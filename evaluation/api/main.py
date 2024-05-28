from datetime import datetime, UTC
import os
from typing import List, Union
from bson import ObjectId
from bson.errors import InvalidId
from uvicorn import Config, Server
from eval import batch_evaluate, evaluate_chat
from fastapi import Query, FastAPI, Response, status
from internal_shared.models.evaluation.database_models import (
    PopulatedRuns,
    Runs,
    Evaluations,
    Iterations,
)
from internal_shared.models.evaluation.models import (
    ChatEvaluationRequest,
    EvaluationRequest,
    EvaluationResponse,
)
from internal_shared.db.mongo import get_async_db

app = FastAPI()

_EVALUATION_DB = "evaluation_db"


@app.get("/")
def ping():
    return {"status": status.HTTP_200_OK}


@app.post(
    "/evaluate",
    summary="Evaluate a dataset",
    description="Evaluates the dataset and optionally evaluates data points multiple times. Saves the results in the database",
    response_description="A type containing all IDs created during the run for the run, evaluations and iterations.",
    response_model=EvaluationResponse,
    response_model_by_alias=False,
)
async def evaluate_dataset(req: EvaluationRequest):
    return await batch_evaluate(req)


@app.post(
    "/evaluate/chat",
    summary="Evaluate a chat session",
    description="Evaluates a chat session and saves the results in the database",
    response_model=Iterations,
    response_model_by_alias=False,
)
async def evaluate_chat_session(req: ChatEvaluationRequest):
    return await evaluate_chat(req)


@app.get(
    "/runs/{run_id}",
    description="Get a specific run",
    response_model=Union[PopulatedRuns, Runs],
    response_model_by_alias=False,
)
async def get_run(
    run_id: str,
    run_type: str = None,
    populate: bool = False,
    from_date: datetime = Query(None),
    until_date: datetime = Query(None),
):
    query = {}

    # apply filter how to look for the run
    try:
        query["_id"] = ObjectId(run_id)
    except (InvalidId, TypeError):
        return Response(
            status_code=status.HTTP_400_BAD_REQUEST, content="Invalid run_id format"
        )

    # apply run_type filter
    if run_type:
        query["run_type"] = run_type

    # Apply date filters
    date_query = {}
    if from_date:
        date_query["$gte"] = from_date
    if until_date:
        date_query["$lte"] = until_date
    else:
        date_query["$lte"] = datetime.now(UTC)

    if date_query:
        query["start_time"] = date_query

    db = await get_async_db(_EVALUATION_DB)
    if db is None:
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content="Database not found",
        )

    run = await db.runs.find_one(query)

    if run is None:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    if populate:
        try:
            evaluations = await db.evaluations.find({"run_id": run["_id"]}).to_list(
                length=None
            )
            for evaluation in evaluations:
                iterations = await db.iterations.find(
                    {"evaluation_id": evaluation["_id"]}
                ).to_list(length=None)
                evaluation["iterations"] = iterations

            run["evaluations"] = evaluations
        except Exception as e:
            return Response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=f"An error occurred: {str(e)}",
            )

    return run


@app.get(
    "/runs/{run_id}/evaluations",
    response_description="Get all evaluations for a specific run",
    response_model=List[Evaluations],
    response_model_by_alias=False,
)
async def get_evaluations(run_id: str, skip: int = 0, limit: int = 100):
    db = await get_async_db(_EVALUATION_DB)
    if db is None:
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content="Database not found",
        )

    evaluations = (
        await db.evaluations.find({"run_id": ObjectId(run_id)})
        .skip(skip)
        .limit(limit)
        .to_list(length=limit)
    )
    return evaluations


@app.get(
    "/evaluations/{evaluation_id}/",
    response_description="Get a specific evaluation by its ID",
    response_model=Evaluations,
    response_model_by_alias=False,
)
async def get_specific_evaluation(evaluation_id: str):
    db = await get_async_db(_EVALUATION_DB)
    if db is None:
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content="Database not found",
        )

    evaluations = await db.evaluations.find_one({"_id": ObjectId(evaluation_id)})
    return evaluations


@app.get(
    "/evaluations/{evaluation_id}/iterations",
    response_description="Get all iterations for a specific evaluation",
    response_model=List[Iterations],
    response_model_by_alias=False,
)
async def get_iterations(evaluation_id: str, skip: int = 0, limit: int = 100):
    db = await get_async_db(_EVALUATION_DB)
    if db is None:
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content="Database not found",
        )

    iterations = (
        await db.iterations.find({"evaluation_id": ObjectId(evaluation_id)})
        .skip(skip)
        .limit(limit)
        .to_list(length=limit)
    )
    return iterations


@app.get(
    "/iterations/{iteration_id}",
    response_description="Get a specific iteration by its ID",
    response_model=Iterations,
    response_model_by_alias=False,
)
async def get_specific_iteration(iteration_id: str):
    db = await get_async_db(_EVALUATION_DB)
    if db is None:
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content="Database not found",
        )

    iterations = await db.iterations.find_one({"_id": ObjectId(iteration_id)})
    return iterations


def main():
    dev_port = os.getenv("EVAL_DEV_PORT")
    if dev_port:
        config = Config(app=app, port=int(dev_port))
    else:
        config = Config(app=app)
    server = Server(config=config)
    server.run()


if __name__ == "__main__":
    main()
