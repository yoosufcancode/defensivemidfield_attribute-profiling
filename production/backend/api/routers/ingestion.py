"""
Stage 1 router — ingestion + utility endpoints.

Pipeline endpoints
  POST /stage1/ingest               trigger ingestion job
  GET  /stage1/status/{job_id}      poll job status

Utility endpoints
  GET  /stage1/teams                list available leagues and teams from existing CSVs
  GET  /stage1/available-data       list already-processed wyscout_{league}_features.csv files
"""
import json
import uuid
import asyncio
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_task_store, get_executor
from api.unicode_utils import fix_unicode, fix_df_unicode
from services.task_store import TaskStore, TaskStatus
from schemas.ingestion import IngestRequest
from config import settings
from workers.pipeline_worker import run_stage1

router = APIRouter(prefix="/stage1", tags=["Stage 1 – Ingestion"])

ALL_LEAGUES = ["Spain", "England", "France", "Germany", "Italy"]


def task_response(record) -> dict:
    """Serialize a TaskRecord to a status response dict."""
    return {
        "job_id": record.job_id,
        "status": record.status,
        "progress": record.progress,
        "message": record.message,
        "result": record.result,
        "error": record.error,
    }


@router.post("/ingest")
async def ingest(
    body: IngestRequest,
    store: TaskStore = Depends(get_task_store),
    executor=Depends(get_executor),
):
    """Enqueue a Stage 1 ingestion job and return its job_id."""
    job_id = str(uuid.uuid4())
    await store.create(job_id)

    asyncio.create_task(
        run_stage1(
            job_id=job_id,
            params={
                "leagues":       body.leagues,
                "skip_download": body.skip_download,
            },
            store=store,
            executor=executor,
        )
    )

    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def ingest_status(job_id: str, store: TaskStore = Depends(get_task_store)):
    """Poll the status of an ingestion job."""
    record = await store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return task_response(record)


@router.get("/teams")
def list_teams():
    """
    Return available leagues and teams from existing wyscout_{league}_features.csv files.
    If CSVs don't exist yet, return leagues list only.
    """
    processed_dir = Path(settings.data_processed_dir)

    leagues_found = []
    teams_by_league: dict[str, list[str]] = {}

    for league in ALL_LEAGUES:
        csv_path = processed_dir / f"wyscout_{league}_features.csv"
        if not csv_path.exists():
            continue
        leagues_found.append(league)
        try:
            df = fix_df_unicode(pd.read_csv(csv_path, usecols=lambda c: c in {"team_name"}))
            if "team_name" in df.columns:
                teams_by_league[league] = sorted(df["team_name"].dropna().unique().tolist())
            else:
                teams_by_league[league] = []
        except Exception:
            teams_by_league[league] = []

    return {
        "leagues": ALL_LEAGUES,
        "leagues_with_data": leagues_found,
        "teams_by_league": teams_by_league,
    }


def _load_match_results(league: str) -> dict:
    """Load matches JSON and return a dict keyed by match_id string with result info."""
    raw_dir = Path(settings.data_raw_dir)
    matches_path = raw_dir / "wyscout" / "matches" / f"matches_{league}.json"
    if not matches_path.exists():
        return {}
    try:
        with open(matches_path) as f:
            matches_data = json.load(f)
        results = {}
        for m in matches_data:
            mid = str(m.get("wyId", ""))
            teams_data = m.get("teamsData", {})
            scores = {}
            sides = {}
            for tid, td in teams_data.items():
                scores[tid] = int(td.get("score", 0))
                sides[tid] = td.get("side", "")
            results[mid] = {
                "winner_id": str(m.get("winner", "")),
                "label":     fix_unicode(m.get("label", "")),
                "date":      m.get("dateutc", "")[:10],
                "scores":    scores,
                "sides":     sides,
            }
        return results
    except Exception:
        return {}


@router.get("/matches")
def list_matches(league: str):
    """Return all unique matches for a league with teams and result info."""
    processed_dir = Path(settings.data_processed_dir)
    csv_path = processed_dir / f"wyscout_{league}_features.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"No data for league '{league}'")
    try:
        df = fix_df_unicode(pd.read_csv(csv_path, usecols=lambda c: c in {"match_id", "team_name", "team_id"}))
        df["base_id"] = df["match_id"].str.replace(r"_P[12]$", "", regex=True)
        results_lookup = _load_match_results(league)

        # Build team_id → team_name mapping
        id_to_name: dict[str, str] = {}
        if "team_id" in df.columns:
            for _, row in df[["team_id", "team_name"]].drop_duplicates().iterrows():
                id_to_name[str(int(row["team_id"]))] = str(row["team_name"])

        matches = []
        for base_id, grp in df.groupby("base_id"):
            teams = sorted(grp["team_name"].dropna().unique().tolist())
            res = results_lookup.get(str(base_id), {})
            winner_id = res.get("winner_id", "")
            winner_name = id_to_name.get(winner_id, "")
            scores_by_name = {id_to_name.get(tid, tid): sc for tid, sc in res.get("scores", {}).items()}
            matches.append({
                "match_id":    str(base_id),
                "teams":       teams,
                "date":        res.get("date", ""),
                "label":       res.get("label", ""),
                "winner":      winner_name,
                "scores":      scores_by_name,
            })
        return sorted(matches, key=lambda x: x["date"] or x["match_id"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


MATCH_STAT_COLS = [
    "bypasses_per_halftime", "passes_attempted", "pass_completion_rate",
    "progressive_passes", "key_passes", "ball_recoveries",
    "tackles_won", "sliding_tackles", "zone14_touches",
    "shot_creating_actions", "possession_time_seconds", "average_position_x",
]


@router.get("/match-detail")
def match_detail(league: str, match_id: str):
    """Return per-player aggregated stats for a single match (both halves averaged)."""
    processed_dir = Path(settings.data_processed_dir)
    csv_path = processed_dir / f"wyscout_{league}_features.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"No data for league '{league}'")
    try:
        use_cols = {"match_id", "team_name", "team_id", "player_name"} | set(MATCH_STAT_COLS)
        df = fix_df_unicode(pd.read_csv(csv_path, usecols=lambda c: c in use_cols))
        df_match = df[df["match_id"].str.startswith(match_id + "_P")]
        if df_match.empty:
            raise HTTPException(status_code=404, detail=f"Match '{match_id}' not found")
        stat_cols = [c for c in MATCH_STAT_COLS if c in df_match.columns]
        agg = (
            df_match.groupby(["team_name", "player_name"])[stat_cols]
            .mean()
            .round(3)
            .reset_index()
        )

        # Compute position bucket using league-wide percentiles
        pos_series = df["average_position_x"].dropna() if "average_position_x" in df.columns else pd.Series(dtype=float)
        q33 = float(pos_series.quantile(0.33)) if not pos_series.empty else 0
        q67 = float(pos_series.quantile(0.67)) if not pos_series.empty else 100

        def _bucket(x):
            if pd.isna(x):
                return "MF"
            return "DM" if x <= q33 else ("CM" if x <= q67 else "AM")

        # Detect per-player mixed positions across the two halves
        if "average_position_x" in df_match.columns:
            df_match = df_match.copy()
            df_match["_period"] = df_match["match_id"].str.extract(r"_P([12])$")[0]
            df_match["_bucket"] = df_match["average_position_x"].map(_bucket)
            mixed = (
                df_match.groupby(["team_name", "player_name"])["_bucket"]
                .nunique()
                .gt(1)
            )
            mixed_set = set(mixed[mixed].index.tolist())
        else:
            mixed_set = set()

        def _assign_bucket(row):
            key = (row["team_name"], row["player_name"])
            if key in mixed_set:
                return "Mixed"
            return _bucket(row.get("average_position_x"))

        agg["position_bucket"] = agg.apply(_assign_bucket, axis=1)
        # Build team_id → team_name mapping for result lookup
        id_to_name: dict[str, str] = {}
        if "team_id" in df.columns:
            for _, row in df[["team_id", "team_name"]].drop_duplicates().iterrows():
                id_to_name[str(int(row["team_id"]))] = str(row["team_name"])

        teams = {}
        for team, grp in agg.groupby("team_name"):
            teams[team] = (
                grp.drop(columns="team_name")
                .sort_values("bypasses_per_halftime", ascending=True)
                .to_dict("records")
            )


        res = _load_match_results(league).get(match_id, {})
        winner_id = res.get("winner_id", "")
        winner_name = id_to_name.get(winner_id, "")
        scores_by_name = {id_to_name.get(tid, tid): sc for tid, sc in res.get("scores", {}).items()}

        return {
            "match_id": match_id,
            "league":   league,
            "teams":    teams,
            "result": {
                "winner": winner_name,
                "scores": scores_by_name,
                "date":   res.get("date", ""),
                "label":  res.get("label", ""),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/available-data")
def available_data():
    """
    Scan data/processed/ for wyscout_{league}_features.csv files and return metadata.
    Returns {league, features_path, row_count} for each found file.
    """
    processed_dir = Path(settings.data_processed_dir)
    if not processed_dir.exists():
        return []

    results: list[dict] = []
    for league in ALL_LEAGUES:
        csv_path = processed_dir / f"wyscout_{league}_features.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in {"player_id"})
            row_count = int(df["player_id"].count()) if "player_id" in df.columns else 0
            unique_players = int(df["player_id"].nunique()) if "player_id" in df.columns else 0
        except Exception:
            row_count = 0
            unique_players = 0

        results.append({
            "league":         league,
            "features_path":  str(csv_path),
            "row_count":      row_count,
            "unique_players": unique_players,
        })

    return results
