"""Stage 4: Per-team model building — MLR, Ridge, Lasso with LOOCV Spearman selection
and scouting gradient extraction matching the notebook approach."""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, Lasso
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import statsmodels.api as sm

OPP_PREFIXES = ("opp_", "score_diff")


def _loocv_spearman(model, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    """Proper LOOCV Spearman ρ — re-fits the model on each fold."""
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    safe_params = {k: v for k, v in model.get_params().items()
                   if k not in ("cv", "store_cv_values", "store_cv_results")}
    for tr_idx, te_idx in loo.split(X):
        m = type(model)(**safe_params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        y_pred[te_idx] = m.predict(X.iloc[te_idx])
    if np.std(y_pred) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    rho, p = spearmanr(y, y_pred)
    return float(rho), float(p)


def _test_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    if np.std(preds) == 0 or np.std(y_test) == 0:
        rho, p = 0.0, 1.0
    else:
        rho, p = spearmanr(y_test, preds)
    r2   = r2_score(y_test, preds) if y_test.var() > 0 else 0.0
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    return {"spearman": float(rho), "spearman_p": float(p),
            "r2": float(r2), "rmse": rmse, "mae": mae}


def run_model_building(
    features_path: str,
    league: str,
    team: str,
    selected_features: list[str],
    target_col: str,
    test_size: float,
    random_state: int,
    progress_cb: Callable[[int, str], None],
) -> dict:
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.player_recommendations import load_half_match, get_feature_cols, prepare_xy

    if not target_col:
        target_col = "bypasses_per_halftime"

    # ── Step 1: Load team-specific half-match rows ────────────────────────────
    progress_cb(5, f"Loading {team} ({league}) match data")
    df_team = load_half_match(features_path, team=team)

    # ── Step 2: Feature selection ─────────────────────────────────────────────
    progress_cb(10, "Selecting features")
    auto_features = get_feature_cols(df_team)

    # Intersect with Stage 3 whitelist if provided, else use auto-detected set
    if selected_features:
        available = [f for f in selected_features if f in auto_features]
    else:
        available = auto_features

    if not available:
        raise ValueError(f"No usable features found for team '{team}'. "
                         "Run Stage 1 ingestion and ensure the team has data.")

    # Exclude opponent-context features — scouting uses player attributes only
    scout_features = [f for f in available
                      if not any(f.startswith(p) for p in OPP_PREFIXES)]

    if len(scout_features) < 3:
        raise ValueError(f"Fewer than 3 scout features available for '{team}'. "
                         "Try adding more leagues or lowering feature-selection thresholds.")

    # ── Step 3: Prepare X/y (scale on scout features only) ───────────────────
    progress_cb(15, "Preparing and scaling data")
    X_sc, y, imp, scaler = prepare_xy(df_team, scout_features, target_col, min_passes=5)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # ── Step 4: Fit all three models ──────────────────────────────────────────
    alphas_ridge = np.logspace(-2, 3, 50)
    alphas_lasso = np.logspace(-4, 1, 50)

    progress_cb(20, "Training MLR")
    mlr = LinearRegression().fit(X_train, y_train)

    progress_cb(28, "Training Ridge (CV)")
    ridge = RidgeCV(alphas=alphas_ridge, cv=5).fit(X_train, y_train)

    progress_cb(36, "Training Lasso (CV)")
    lasso_cv = LassoCV(alphas=alphas_lasso, cv=5, max_iter=2000, random_state=42).fit(X_train, y_train)
    # If CV selected the ceiling alpha, all coefficients are zero — fall back to the
    # lowest alpha so Lasso still produces meaningful predictions for this dataset.
    if lasso_cv.alpha_ >= alphas_lasso[-1] * 0.99:
        lasso = Lasso(alpha=float(alphas_lasso[0]), max_iter=2000).fit(X_train, y_train)
    else:
        lasso = lasso_cv

    models_map = {"MLR": mlr, "Ridge": ridge, "Lasso": lasso}

    # ── Step 5: LOOCV Spearman ρ on training set (model selection metric) ─────
    progress_cb(44, "LOOCV evaluation — MLR")
    rho_mlr,   p_mlr   = _loocv_spearman(mlr,   X_train, y_train)

    progress_cb(52, "LOOCV evaluation — Ridge")
    rho_ridge, p_ridge = _loocv_spearman(ridge, X_train, y_train)

    progress_cb(60, "LOOCV evaluation — Lasso")
    rho_lasso, p_lasso = _loocv_spearman(lasso, X_train, y_train)

    loocv_rho = {"MLR": (rho_mlr, p_mlr), "Ridge": (rho_ridge, p_ridge), "Lasso": (rho_lasso, p_lasso)}
    best_name  = max(loocv_rho, key=lambda k: loocv_rho[k][0])
    best_model = models_map[best_name]

    # ── Step 6: Test-set metrics for all models ───────────────────────────────
    progress_cb(65, "Test-set evaluation")
    results = []
    file_stems = {"MLR": "mlr_model", "Ridge": "ridge_model", "Lasso": "lasso_model"}

    for name, m in models_map.items():
        rho_cv, p_cv = loocv_rho[name]
        tm = _test_metrics(m, X_test, y_test)
        results.append({
            "name": name,
            "loocv": {
                "spearman":   round(rho_cv, 4),
                "spearman_p": round(p_cv, 4),
                "r2":         tm["r2"],     # test-set R² for display
                "rmse":       tm["rmse"],
                "mae":        tm["mae"],
            },
            "test": {k: round(v, 4) for k, v in tm.items()},
            "model_path": "",               # filled after saving
        })

    # ── Step 7: OLS p-values + 5-fold sign stability → scouting gradients ─────
    progress_cb(70, "Extracting scouting gradients")
    X_sm = sm.add_constant(X_train, has_constant="add")
    ols  = sm.OLS(y_train, X_sm).fit()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_coefs = []
    for tr_idx, _ in kf.split(X_train):
        if best_name == "MLR":
            m_fold = LinearRegression()
        elif best_name == "Ridge":
            m_fold = Ridge(alpha=float(best_model.alpha_))
        else:
            m_fold = Lasso(alpha=float(getattr(best_model, "alpha_", best_model.alpha)), max_iter=2000)
        m_fold.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        fold_coefs.append(m_fold.coef_)

    fold_coefs = np.array(fold_coefs)
    mean_coefs = fold_coefs.mean(axis=0)

    rows = []
    scouting_grads = {}
    for i, fname in enumerate(scout_features):
        signs       = np.sign(fold_coefs[:, i])
        sign_stable = len(set(signs)) == 1
        mean_b      = float(mean_coefs[i])
        pval        = float(ols.pvalues.get(fname, 1.0))
        if sign_stable and pval < 0.15:
            scouting_grads[fname] = mean_b
        rows.append({"feature": fname, "mean_coef": mean_b,
                     "p_value": pval, "sign_stable": sign_stable})

    # Fallback — ensure at least 6 scouting features
    MIN_SCOUT = 6
    if len(scouting_grads) < MIN_SCOUT:
        needed = MIN_SCOUT - len(scouting_grads)
        extras = sorted(
            [r for r in rows if r["feature"] not in scouting_grads],
            key=lambda r: (0 if r["sign_stable"] else 1, r["p_value"]),
        )
        for r in extras[:needed]:
            scouting_grads[r["feature"]] = r["mean_coef"]

    # Build structured scouting_features list
    total_abs = sum(abs(v) for v in scouting_grads.values()) or 1.0
    scouting_features_list = []
    for fname, mean_b in sorted(scouting_grads.items(), key=lambda x: abs(x[1]), reverse=True):
        r = next(r for r in rows if r["feature"] == fname)
        pval = r["p_value"]
        tier = ("Tier 1 — high confidence" if pval < 0.05 else
                "Tier 2 — moderate confidence" if pval < 0.10 else
                "Tier 3 — indicative only"    if pval < 0.15 else
                "Tier 4 — fallback only")
        scouting_features_list.append({
            "feature":         fname,
            "gradient":        float(mean_b),
            "direction":       "look for LOW" if mean_b > 0 else "look for HIGH",
            "p_value":         float(pval),
            "sign_stable":     bool(r["sign_stable"]),
            "confidence_tier": tier,
        })

    # ── Step 8: Save per-team models and scaler ───────────────────────────────
    progress_cb(85, "Saving models")
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{league}_{team.replace(' ', '_')}"

    scaler_path = models_dir / f"{stem}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    best_model_path = ""
    for res_entry in results:
        name      = res_entry["name"]
        path      = models_dir / f"{stem}_{file_stems[name]}.pkl"
        with open(path, "wb") as f:
            pickle.dump(models_map[name], f)
        res_entry["model_path"] = str(path)
        if name == best_name:
            best_model_path = str(path)

    best_entry  = next(r for r in results if r["name"] == best_name)
    sp_test     = best_entry["test"]["spearman"]
    sp_train    = best_entry["loocv"]["spearman"]

    progress_cb(100, "Model building complete")

    return {
        "models":            results,
        "feature_count":     len(scout_features),
        "best_model":        best_name,
        "best_model_path":   best_model_path,
        "scaler_path":       str(scaler_path),
        "available_features": scout_features,
        "scouting_grads":    scouting_grads,
        "scouting_features": scouting_features_list,
        "spearman_test":     round(sp_test, 4),
        "spearman_train":    round(sp_train, 4),
        "league":            league,
        "team":              team,
    }
