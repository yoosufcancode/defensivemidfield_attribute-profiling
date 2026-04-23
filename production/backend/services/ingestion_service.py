"""Stage 1: Data ingestion and feature engineering via Wyscout open dataset."""
import re
import sys
from typing import Callable

_ESCAPE_RE = re.compile(r'\\u([0-9a-fA-F]{4})')

def _decode_unicode_escapes(s):
    if not isinstance(s, str):
        return s
    return _ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), s)


def _decode_player_names(df):
    """Decode literal \\uXXXX sequences in string columns that Wyscout stores as plain text."""
    df = df.copy()
    for col in ("player_name", "team_name"):
        if col in df.columns:
            df[col] = df[col].map(_decode_unicode_escapes)
    return df

ALL_LEAGUES = ["Spain", "England", "France", "Germany", "Italy"]


def run_ingestion(
    leagues: list,
    skip_download: bool,
    progress_cb: Callable[[int, str], None],
) -> dict:
    """
    For each league, call compute_features_for_competition from
    src.features.main_feature. Skips leagues whose CSV already exists when
    skip_download is True.

    Returns dict matching IngestResult schema:
      {
        "features_paths": {league: path_str, ...},
        "row_counts":     {league: n_rows, ...},
        "leagues_processed": [...],
      }
    Raises RuntimeError if no leagues were processed successfully.
    """
    from config import settings

    repo_root = settings.data_raw_dir.parent.parent  # data/raw -> data -> repo root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.features.main_feature import compute_features_for_competition

    processed_dir = settings.data_processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    features_paths: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    unique_player_counts: dict[str, int] = {}
    leagues_processed: list[str] = []
    errors: list[str] = []

    total = len(leagues)
    for i, league in enumerate(leagues):
        base_pct = int(i / total * 90)
        progress_cb(base_pct, f"Processing league: {league}")

        csv_path = processed_dir / f"wyscout_{league}_features.csv"

        if skip_download and csv_path.exists():
            progress_cb(base_pct + int(90 / total * 0.9), f"Skipping {league} — CSV already exists")
            import pandas as pd
            try:
                df_existing = pd.read_csv(csv_path, usecols=lambda c: c in {"player_id"})
                n = int(df_existing["player_id"].count()) if "player_id" in df_existing.columns else 0
                u = int(df_existing["player_id"].nunique()) if "player_id" in df_existing.columns else 0
            except Exception:
                n = 0
                u = 0
            features_paths[league] = str(csv_path)
            row_counts[league] = n
            unique_player_counts[league] = u
            leagues_processed.append(league)
            continue

        try:
            progress_cb(base_pct + 2, f"Computing features for {league}...")
            df = compute_features_for_competition(league=league)
            if df is not None and not df.empty:
                df = _decode_player_names(df)
                df.to_csv(csv_path, index=False)
                features_paths[league] = str(csv_path)
                row_counts[league] = len(df)
                unique_player_counts[league] = int(df["player_id"].nunique()) if "player_id" in df.columns else 0
                leagues_processed.append(league)
                progress_cb(base_pct + int(90 / total * 0.95), f"Done {league}: {len(df)} rows")
            else:
                msg = f"{league}: feature engineering returned no data"
                errors.append(msg)
                progress_cb(base_pct + int(90 / total * 0.95), f"Warning: {msg}")
        except Exception as exc:
            msg = f"{league}: {exc}"
            errors.append(msg)
            progress_cb(base_pct + int(90 / total * 0.95), f"Error — {msg}")

    if not leagues_processed:
        raise RuntimeError(
            "No leagues were processed successfully.\n" + "\n".join(errors)
        )

    progress_cb(100, f"Ingestion complete — {len(leagues_processed)} league(s) processed")

    return {
        "features_paths": features_paths,
        "row_counts": row_counts,
        "unique_player_counts": unique_player_counts,
        "leagues_processed": leagues_processed,
    }
