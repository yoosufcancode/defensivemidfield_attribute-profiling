from pydantic import BaseModel
from typing import Any


class FeatureSelectionRequest(BaseModel):
    features_path: str
    team: str                              # filter to this team's half-match rows
    target_col: str = "bypasses_per_halftime"
    n_top: int = 15


class FeatureSelectionResult(BaseModel):
    univariate: list[dict[str, Any]]
    mutual_info: list[dict[str, Any]]
    random_forest: list[dict[str, Any]]
    rfe: list[dict[str, Any]]
    consensus: list[dict[str, Any]]
    selected_features: list[str]
