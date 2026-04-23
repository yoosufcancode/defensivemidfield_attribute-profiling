"""
Asyncio bridge between FastAPI and CPU-bound service functions.

Each `run_stageN` coroutine:
  1. Marks the job as running.
  2. Offloads the blocking call to a ProcessPoolExecutor via run_in_executor.
  3. Progress callbacks from the worker thread post back through
     asyncio.run_coroutine_threadsafe so they are safe to call from any thread.
  4. Marks the job completed or failed.
"""
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from services.task_store import TaskStore, TaskStatus


def make_progress_cb(job_id: str, store: TaskStore, loop: asyncio.AbstractEventLoop):
    """Return a thread-safe progress callback."""
    def cb(progress: int, message: str) -> None:
        """Post a progress update to the task store from any thread."""
        store.update_sync(job_id, loop, progress=progress, message=message)
    return cb


def ingestion_worker(leagues, skip_download, job_id, store, loop):
    """Worker entry point for Stage 1 ingestion, called in a subprocess."""
    from services.ingestion_service import run_ingestion
    cb = make_progress_cb(job_id, store, loop)
    return run_ingestion(leagues, skip_download, cb)


async def run_stage1(
    job_id: str,
    params: dict,
    store: TaskStore,
    executor: ThreadPoolExecutor,
) -> None:
    """Orchestrate Stage 1 ingestion: offload to executor and update task store on completion."""
    loop = asyncio.get_running_loop()
    await store.update(job_id, status=TaskStatus.running, progress=0, message="Starting ingestion")
    try:
        result = await loop.run_in_executor(
            executor,
            ingestion_worker,
            params["leagues"],
            params["skip_download"],
            job_id,
            store,
            loop,
        )
        await store.update(
            job_id,
            status=TaskStatus.completed,
            progress=100,
            message="Done",
            result=result,
        )
    except Exception as exc:
        await store.update(
            job_id,
            status=TaskStatus.failed,
            message=str(exc),
            error=traceback.format_exc(),
        )


def eda_worker(features_path, job_id, store, loop):
    """Worker entry point for Stage 2 EDA, called in a subprocess."""
    from services.eda_service import run_eda
    cb = make_progress_cb(job_id, store, loop)
    return run_eda(features_path, cb)


async def run_stage2(
    job_id: str,
    params: dict,
    store: TaskStore,
    executor: ThreadPoolExecutor,
) -> None:
    """Orchestrate Stage 2 EDA: offload to executor and update task store on completion."""
    loop = asyncio.get_running_loop()
    await store.update(job_id, status=TaskStatus.running, progress=0, message="Starting EDA")
    try:
        result = await loop.run_in_executor(
            executor,
            eda_worker,
            params["features_path"],
            job_id,
            store,
            loop,
        )
        await store.update(
            job_id,
            status=TaskStatus.completed,
            progress=100,
            message="Done",
            result=result,
        )
    except Exception as exc:
        await store.update(
            job_id,
            status=TaskStatus.failed,
            message=str(exc),
            error=traceback.format_exc(),
        )


def feature_selection_worker(features_path, team, target_col, n_top, job_id, store, loop):
    """Worker entry point for Stage 3 feature selection, called in a subprocess."""
    from services.feature_selection_service import run_feature_selection
    cb = make_progress_cb(job_id, store, loop)
    return run_feature_selection(features_path, team, target_col, n_top, cb)


async def run_stage3(
    job_id: str,
    params: dict,
    store: TaskStore,
    executor: ThreadPoolExecutor,
) -> None:
    """Orchestrate Stage 3 feature selection: offload to executor and update task store on completion."""
    loop = asyncio.get_running_loop()
    await store.update(job_id, status=TaskStatus.running, progress=0, message="Starting feature selection")
    try:
        result = await loop.run_in_executor(
            executor,
            feature_selection_worker,
            params["features_path"],
            params["team"],
            params["target_col"],
            params["n_top"],
            job_id,
            store,
            loop,
        )
        await store.update(
            job_id,
            status=TaskStatus.completed,
            progress=100,
            message="Done",
            result=result,
        )
    except Exception as exc:
        await store.update(
            job_id,
            status=TaskStatus.failed,
            message=str(exc),
            error=traceback.format_exc(),
        )


def model_building_worker(features_path, league, team, selected_features, target_col, test_size, random_state, job_id, store, loop):
    """Worker entry point for Stage 4 model building, called in a subprocess."""
    from services.model_service import run_model_building
    cb = make_progress_cb(job_id, store, loop)
    return run_model_building(features_path, league, team, selected_features, target_col, test_size, random_state, cb)


async def run_stage4(
    job_id: str,
    params: dict,
    store: TaskStore,
    executor: ThreadPoolExecutor,
) -> None:
    """Orchestrate Stage 4 model building: offload to executor and update task store on completion."""
    loop = asyncio.get_running_loop()
    await store.update(job_id, status=TaskStatus.running, progress=0, message="Starting model building")
    try:
        result = await loop.run_in_executor(
            executor,
            model_building_worker,
            params["features_path"],
            params["league"],
            params["team"],
            params["selected_features"],
            params["target_col"],
            params["test_size"],
            params["random_state"],
            job_id,
            store,
            loop,
        )
        await store.update(
            job_id,
            status=TaskStatus.completed,
            progress=100,
            message="Done",
            result=result,
        )
    except Exception as exc:
        await store.update(
            job_id,
            status=TaskStatus.failed,
            message=str(exc),
            error=traceback.format_exc(),
        )


def replacement_worker(league, team, top_n, min_matches,
                       bypass_ceiling_percentile,
                       scouting_grads, scouting_features, model_selected,
                       spearman_test, spearman_train,
                       job_id, store, loop):
    """Worker entry point for Stage 6 replacement analysis, called in a subprocess."""
    from services.replacement_service import run_replacement_analysis
    cb = make_progress_cb(job_id, store, loop)
    return run_replacement_analysis(
        league, team, top_n, min_matches,
        bypass_ceiling_percentile=bypass_ceiling_percentile,
        scouting_grads=scouting_grads,
        scouting_features=scouting_features,
        model_selected=model_selected,
        spearman_test=spearman_test,
        spearman_train=spearman_train,
        progress_cb=cb,
    )


async def run_stage6(
    job_id: str,
    params: dict,
    store: TaskStore,
    executor,
) -> None:
    """Orchestrate Stage 6 replacement analysis: offload to executor and update task store on completion."""
    loop = asyncio.get_running_loop()
    await store.update(job_id, status=TaskStatus.running, progress=0, message="Starting replacement analysis")
    try:
        result = await loop.run_in_executor(
            executor,
            replacement_worker,
            params["league"],
            params["team"],
            params["top_n"],
            params["min_matches"],
            params.get("bypass_ceiling_percentile"),
            params.get("scouting_grads"),
            params.get("scouting_features"),
            params.get("model_selected", ""),
            params.get("spearman_test", 0.0),
            params.get("spearman_train", 0.0),
            job_id,
            store,
            loop,
        )
        await store.update(
            job_id,
            status=TaskStatus.completed,
            progress=100,
            message="Done",
            result=result,
        )
    except Exception as exc:
        await store.update(
            job_id,
            status=TaskStatus.failed,
            message=str(exc),
            error=traceback.format_exc(),
        )


def best_model_worker(model_path, scaler_path, features_path, selected_features, target_col, job_id, store, loop):
    """Worker entry point for Stage 5 best-model analysis, called in a subprocess."""
    from services.best_model_service import run_best_model_analysis
    cb = make_progress_cb(job_id, store, loop)
    return run_best_model_analysis(model_path, scaler_path, features_path, selected_features, target_col, cb)


async def run_stage5(
    job_id: str,
    params: dict,
    store: TaskStore,
    executor: ThreadPoolExecutor,
) -> None:
    """Orchestrate Stage 5 best-model analysis: offload to executor and update task store on completion."""
    loop = asyncio.get_running_loop()
    await store.update(job_id, status=TaskStatus.running, progress=0, message="Starting best model analysis")
    try:
        result = await loop.run_in_executor(
            executor,
            best_model_worker,
            params["model_path"],
            params["scaler_path"],
            params["features_path"],
            params["selected_features"],
            params["target_col"],
            job_id,
            store,
            loop,
        )
        await store.update(
            job_id,
            status=TaskStatus.completed,
            progress=100,
            message="Done",
            result=result,
        )
    except Exception as exc:
        await store.update(
            job_id,
            status=TaskStatus.failed,
            message=str(exc),
            error=traceback.format_exc(),
        )
