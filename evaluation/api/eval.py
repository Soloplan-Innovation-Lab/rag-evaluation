import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt
from typing import Dict, List
from bson import ObjectId
from datasets import Dataset
from deepeval import evaluate as d_evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import TestResult
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    GEval,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from ragas import evaluate as r_evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_similarity,
)
from internal_shared.models.ai.evaluation_models import (
    azure_openai,
    azure_model,
    azure_embeddings,
)
from internal_shared.models.evaluation.database_models import (
    DeepEvalMetric,
    _Evaluations,
    _Iterations,
    Iterations,
    Runs,
)
from internal_shared.models.evaluation.models import (
    ChatEvaluationRequest,
    EvaluationRequest,
    EvaluationPayload,
    EvaluationResponse,
    EvaluationResult,
)
from internal_shared.db.mongo import get_async_db, get_sync_db
from internal_shared.logger import get_logger

_EVALUATION_DB = "evaluation_db"

logger = get_logger(__name__)


async def batch_evaluate(req: EvaluationRequest):
    """
    Evaluates a batch of items using deepeval and RAGAS.
    """
    items = req.dataset
    iterations = req.iterations_per_entry

    db = await get_async_db(_EVALUATION_DB)
    if db is None:
        raise ValueError("Database not available")

    # start run
    start_time = dt.now(datetime.UTC)

    run = Runs(
        run_type=req.run_type,
        description=req.description,
        start_time=start_time,
        total_data_points=len(items),
    )
    run_dict = run.model_dump(by_alias=True, exclude=["id"])
    insert_result = await db.runs.insert_one(run_dict)
    run_id = insert_result.inserted_id

    logger.info(f"Created run with ID {run_id}")

    # evaluate in a separate thread to get another event loop
    with ThreadPoolExecutor() as executor:
        future = executor.submit(_internal_batch_evaluate, items, iterations, run_id)
        batch_result = future.result()

    # Update run end time
    end_time = dt.now(datetime.UTC)
    await db.runs.update_one({"_id": run_id}, {"$set": {"end_time": end_time}})

    logger.info(f"Finished run with ID {run_id}")

    return EvaluationResponse(
        run_id=str(run_id),
        evaluation_ids=batch_result,
    )


async def evaluate_chat(req: ChatEvaluationRequest):
    db = await get_async_db(_EVALUATION_DB)
    if db is None:
        raise ValueError("Database not available")

    start_time = dt.now(datetime.UTC)
    run = Runs(
        run_type=req.run_type,
        description=req.description,
        start_time=start_time,
        total_data_points=1,
        chat_session_id=req.chat_session_id,
    )
    run_dict = run.model_dump(by_alias=True, exclude=["id"])
    insert_result = await db.runs.insert_one(run_dict)
    run_id = insert_result.inserted_id

    logger.info(f"Created run with ID {run_id}")

    with ThreadPoolExecutor() as executor:
        future = executor.submit(_internal_chat_evaluate, req, run_id)
        chat_result = future.result()

    end_time = dt.now(datetime.UTC)
    await db.runs.update_one({"_id": run_id}, {"$set": {"end_time": end_time}})

    logger.info(f"Finished run with ID {run_id}")

    return chat_result


def _internal_chat_evaluate(req: ChatEvaluationRequest, run_id: str):
    db = get_sync_db(_EVALUATION_DB)
    if db is None:
        raise ValueError("Database not available")

    test_case = LLMTestCase(
        input=req.input,
        actual_output=req.actual_output,
        retrieval_context=req.retrieval_context,
    )

    dataset = EvaluationDataset(test_cases=[test_case])
    metrics = [
        _create_deepeval_metric(AnswerRelevancyMetric, 0.7),
        _create_deepeval_metric(FaithfulnessMetric, 0.7, include_reason=True),
        _create_deepeval_metric(ContextualRelevancyMetric, 0.7, include_reason=True),
    ]

    evaluation = _Evaluations(
        run_id=ObjectId(run_id),
        total_iterations=1,
        input=req.input,
        actual_output=req.actual_output,
        retrieval_context=req.retrieval_context,
    )
    evaluation_dict = evaluation.model_dump(by_alias=True, exclude=["id"])
    evaluation_insert_result = db.evaluations.insert_one(evaluation_dict)

    logger.info(f"Created evaluation with ID {evaluation_insert_result.inserted_id}")

    results = d_evaluate(dataset, metrics)

    iteration_data = _Iterations(
        evaluation_id=ObjectId(evaluation_insert_result.inserted_id),
        iteration=0,
        deepeval=_map_deepeval_results(results),
    )
    iteration_dict = iteration_data.model_dump(by_alias=True, exclude=["id"])
    iteration_insert_result = db.iterations.insert_one(iteration_dict)

    logger.info(f"Created iteration with ID {iteration_insert_result.inserted_id}")

    return Iterations(
        id=str(iteration_insert_result.inserted_id),
        evaluation_id=str(evaluation_insert_result.inserted_id),
        iteration=0,
        deepeval=iteration_data.deepeval,
    )


def _internal_batch_evaluate(
    items: List[EvaluationPayload], iterations: int, run_id: str
):
    db = get_sync_db(_EVALUATION_DB)
    if db is None:
        raise ValueError("Database not available")

    evaluation_ids: List[EvaluationResult] = []
    # iterate over the dataset
    for index, data_point in enumerate(items):
        logger.info(f"Processing data point {index + 1}/{len(items)}")

        # store evaluation data
        # since this is the parent of the iteration, we need to store the data point first
        evaluation = _Evaluations(
            run_id=ObjectId(run_id),
            total_iterations=iterations,
            input=data_point.input,
            actual_output=data_point.actual_output,
            expected_output=data_point.expected_output,
            context=data_point.context,
            retrieval_context=data_point.retrieval_context,
        )
        evaluation_dict = evaluation.model_dump(by_alias=True, exclude=["id"])
        evaluation_insert_result = db.evaluations.insert_one(evaluation_dict)

        logger.info(
            f"Created evaluation with ID {evaluation_insert_result.inserted_id}"
        )

        iteration_ids = []
        # evaluate the data points for the given number of iterations
        for iteration in range(iterations):
            logger.info(f"Processing iteration {iteration + 1}/{iterations}")

            deepeval_results = batch_evaluate_deepeval([data_point])
            ragas_results = batch_evaluate_ragas([data_point])

            # store iteration data
            # this is done for each iteration within the evaluation
            iteration_data = _Iterations(
                evaluation_id=ObjectId(evaluation_insert_result.inserted_id),
                iteration=iteration,
                deepeval=_map_deepeval_results(deepeval_results),
                ragas=_map_ragas_results(ragas_results),
            )
            iteration_dict = iteration_data.model_dump(by_alias=True, exclude=["id"])
            iteration_insert_result = db.iterations.insert_one(iteration_dict)

            logger.info(
                f"Created iteration with ID {iteration_insert_result.inserted_id}"
            )
            iteration_ids.append(str(iteration_insert_result.inserted_id))

        # update the evaluation with the iteration ids
        evaluation_ids.append(
            EvaluationResult(
                evaluation_batch_number=index + 1,
                evaluation_id=str(evaluation_insert_result.inserted_id),
                iteration_ids=iteration_ids,
            )
        )

    return evaluation_ids


def batch_evaluate_deepeval(items: list[EvaluationPayload]) -> List[TestResult]:
    """
    This setup and the evaluated metrics are based on the deepeval documentation https://docs.confident-ai.com/docs/guides-rag-evaluation

    These metrics are evaluated:
    - AnswerRelevancyMetric: evaluates whether the prompt template in your generator is able to instruct your LLM to output relevant and helpful outputs based on the retrieval_context.
    - FaithfulnessMetric: evaluates whether the LLM used in your generator can output information that does not hallucinate AND contradict any factual information presented in the retrieval_context.
    - ContextualPrecisionMetric: evaluates whether the reranker in your retriever ranks more relevant nodes in your retrieval context higher than irrelevant ones.
    - ContextualRecallMetric: evaluates whether the embedding model in your retriever is able to accurately capture and retrieve relevant information based on the context of the input.
    - ContextualRelevancyMetric: evaluates whether the text chunk size and top-K of your retriever is able to retrieve information without much irrelevancies.

    In order to actually validate the input, the metrics require some kind of expected output (the teacher value). All other inputs, like the prompt, documents and output are provided by the actual pipeline.

    Note, that these metrics are very generic and simply based on math and statistics. For more domain-specific metrics, approaches like GEval (https://docs.confident-ai.com/docs/metrics-llm-evals, https://arxiv.org/pdf/2303.16634) or SemScore (https://arxiv.org/pdf/2401.17072) can be used.
    """
    test_cases = [_create_deepeval_test_case(item) for item in items]
    dataset = EvaluationDataset(test_cases=test_cases)
    metrics = [
        _create_deepeval_metric(AnswerRelevancyMetric, 0.7),
        _create_deepeval_metric(FaithfulnessMetric, 0.7, include_reason=True),
        _create_deepeval_metric(ContextualPrecisionMetric, 0.7, include_reason=True),
        _create_deepeval_metric(ContextualRecallMetric, 0.7, include_reason=True),
        _create_deepeval_metric(ContextualRelevancyMetric, 0.7, include_reason=True),
    ]
    return d_evaluate(dataset, metrics)


def batch_evaluate_ragas(items: list[EvaluationPayload]):
    """
    Evaluates a Dataset of items using RAGAS and is based on the ragas documentation https://github.com/explodinggradients/ragas

    These metrics are evaluated:
    - faithfulness: measures the factual consistency of the generated answer against the given context.
    - answer_correctness: the assessment involves gauging the accuracy of the generated answer when compared to the ground truth. This evaluation relies on the ground truth and the answer.
    - answer_relevancy: focuses on assessing how pertinent the generated answer is to the given prompt..
    - context_precision: evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not.
    - context_recall: measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth.
    - context_entity_recall: gives the measure of recall of the retrieved context, based on the number of entities present in both ground_truths and contexts relative to the number of entities present in the ground_truths alone.
    - answer_similarity: assessment of the semantic resemblance between the generated answer and the ground truth.

    The ragas score divides these metrics into different sections:
    - generation: faithfulness, answer_correctness, answer_relevancy
    - retrieval: context_precision, context_recall, context_entity_recall

    Thus, the most important metrics are faithfulness, answer_relevancy, context_precision and context_recall. For end-to-end evaluation, the answer_correctness and answer_similarity are also important.
    """
    # Prepare data for RAGAS evaluation
    data_samples = {
        "question": [item.input for item in items],
        "answer": [item.actual_output for item in items],
        "contexts": [[",".join(item.retrieval_context) for item in items]],
        "ground_truth": [item.expected_output for item in items],
    }
    ragas_dataset = Dataset.from_dict(data_samples)

    # Evaluate using RAGAS
    ragas_results = (
        r_evaluate(
            ragas_dataset,
            metrics=[
                faithfulness,
                answer_correctness,
                answer_relevancy,
                context_precision,
                context_recall,
                context_entity_recall,
                answer_similarity,
            ],
            llm=azure_model,
            embeddings=azure_embeddings,
        )
        .to_pandas()
        .to_json(orient="records")
    )
    return json.loads(ragas_results)


def _create_deepeval_test_case(item: EvaluationPayload) -> LLMTestCase:
    """
    Creates a test case for the deepeval evaluation.

    LLMTestCase parameter description (https://docs.confident-ai.com/docs/evaluation-test-cases):
    - input: the actual user input
    - actual_output: the actual output from the model
    - retrieval_context: the documents used for answer generation
    - expected_output: the expected output from the model (provided by the Dataset)
    - context: the ideal retrieval results for a given input (provided by the Dataset)
    """
    return LLMTestCase(
        input=item.input,
        actual_output=item.actual_output,
        retrieval_context=item.retrieval_context,
        expected_output=item.expected_output,
    )


def _create_deepeval_metric(
    metric_class: BaseMetric, threshold: float, **kwargs
) -> BaseMetric:
    """
    Instantiates a deepeval metric with the given threshold and additional keyword arguments.
    """
    return metric_class(threshold=threshold, model=azure_openai, **kwargs)


def _map_deepeval_results(
    deepeval_results: List[TestResult],
) -> Dict[str, DeepEvalMetric]:
    """
    Maps the deepeval results to a dictionary with metric names as keys.
    """
    mapped_deepeval = {}
    for deepeval in deepeval_results[0].metrics:
        d_m = DeepEvalMetric(
            score=deepeval.score,
            reason=deepeval.reason if hasattr(deepeval, "reason") else None,
            success=deepeval.success if hasattr(deepeval, "success") else None,
            threshold=deepeval.threshold,
        ).model_dump()
        mapped_deepeval[deepeval.__name__] = d_m
    return mapped_deepeval


def _map_ragas_results(ragas_results) -> Dict[str, float]:
    """
    Maps the ragas results to a dictionary with metric names as keys.
    """
    mapped_ragas = {}
    for key, value in ragas_results[0].items():
        if key not in ["question", "answer", "contexts", "ground_truth"]:
            mapped_ragas[key] = value
    return mapped_ragas
