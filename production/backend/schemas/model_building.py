from pydantic import BaseModel
from typing import Any, Optional


class ModelBuildRequest(BaseModel):
    features_path: str
    league: str                               # team's league (for file naming)
    team: str                                 # filter data to this team
    selected_features: list[str] = []         # optional whitelist from Stage 3
    target_col: str = "bypasses_per_halftime"
    test_size: float = 0.15
    random_state: int = 42


class ModelMetrics(BaseModel):
    spearman: float
    spearman_p: float
    r2: float
    rmse: float
    mae: float


class ModelResult(BaseModel):
    name: str
    loocv: ModelMetrics
    test: ModelMetrics
    model_path: str = ""


class ModelBuildResult(BaseModel):
    models: list[ModelResult]
    feature_count: int
    best_model: str
    best_model_path: str = ""
    scaler_path: str
    available_features: list[str] = []    # scout features actually used in the model
    scouting_grads: dict = {}             # gradient weights for cross-league scoring
    scouting_features: list = []          # scouting features with p-values / tiers
    spearman_test: float = 0.0
    spearman_train: float = 0.0           # LOOCV Spearman on training set
    league: str = ""
    team: str = ""
