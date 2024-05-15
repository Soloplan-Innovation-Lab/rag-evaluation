import json
from concurrent.futures import ThreadPoolExecutor
from typing import List
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
from models import EvaluationPayload, azure_openai, azure_embeddings, azure_model


def batch_evaluate(items: list[EvaluationPayload]):
    """
    Evaluates a batch of items using deepeval and RAGAS.
    """
    # evaluate in a separate thread to get another event loop
    with ThreadPoolExecutor() as executor:
        future = executor.submit(_internal_batch_evaluate, items)
        deepeval_results = future.result()  # This blocks until the result is available
    return deepeval_results


def _internal_batch_evaluate(items: list[EvaluationPayload]):
    deepeval_results = batch_evaluate_deepeval(items)
    ragas_results = batch_evaluate_ragas(items)
    # probably add ARES? https://arxiv.org/pdf/2311.09476

    # Aggregate results for each test case
    results = []
    for index, item in enumerate(items):
        aggregated_results = {
            "deepeval": deepeval_results[index],
            "ragas": ragas_results[index],
        }
        results.append(aggregated_results)

    return results


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
        "question": [item.prompt for item in items],
        "answer": [item.output for item in items],
        "contexts": [[",".join(item.documents) for item in items]],
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
        input=item.prompt,
        actual_output=item.output,
        retrieval_context=item.documents,
        expected_output=item.expected_output,
    )


def _create_deepeval_metric(
    metric_class: BaseMetric, threshold: float, **kwargs
) -> BaseMetric:
    """
    Instantiates a deepeval metric with the given threshold and additional keyword arguments.
    """
    return metric_class(threshold=threshold, model=azure_openai, **kwargs)
