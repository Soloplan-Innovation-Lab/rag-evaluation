import json
from deepeval import evaluate as d_evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    GEval,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from models import EvaluationPayload, azure_openai, azure_embeddings, azure_model
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
from datasets import Dataset


def evaluate(item: EvaluationPayload):
    evaluation_dict = {}

    # deepeval evaluation
    test_case = LLMTestCase(
        input=item.prompt,
        actual_output=item.output,
        retrieval_context=item.documents,
        expected_output=item.expected_output,
    )

    evaluation_dict["deepeval_g_eval"] = deepeval_g_eval(test_case)
    evaluation_dict["deepeval_evaluate_answer_relevancy"] = (
        deepeval_evaluate_answer_relevancy(test_case)
    )
    evaluation_dict["deepeval_evaluate_faithfulness"] = deepeval_evaluate_faithfulness(
        test_case
    )
    evaluation_dict["deepeval_evaluate_contextual_precision"] = (
        deepeval_evaluate_contextual_precision(test_case)
    )
    evaluation_dict["deepeval_evaluate_contextual_recall"] = (
        deepeval_evaluate_contextual_recall(test_case)
    )
    evaluation_dict["deepeval_evaluate_contextual_relevancy"] = (
        deepeval_evaluate_contextual_relevancy(test_case)
    )
    evaluation_dict["deepeval_evaluate_hallucination"] = (
        deepeval_evaluate_hallucination(item)
    )

    # ragas evaluation
    evaluation_dict["ragas_evaluate"] = json.loads(ragas_evaluate(item))
    return evaluation_dict

# ref: https://docs.confident-ai.com/docs/getting-started
def deepeval_evaluate_answer_relevancy(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=azure_openai)
    return d_evaluate([test_case], [answer_relevancy_metric])


def deepeval_evaluate_faithfulness(test_case: LLMTestCase):
    metric = FaithfulnessMetric(threshold=0.7, model=azure_openai, include_reason=True)
    return d_evaluate([test_case], [metric])


def deepeval_evaluate_contextual_precision(test_case: LLMTestCase):
    metric = ContextualPrecisionMetric(
        threshold=0.7, model=azure_openai, include_reason=True
    )
    return d_evaluate([test_case], [metric])


def deepeval_evaluate_contextual_recall(test_case: LLMTestCase):
    metric = ContextualRecallMetric(
        threshold=0.7, model=azure_openai, include_reason=True
    )
    return d_evaluate([test_case], [metric])


def deepeval_evaluate_contextual_relevancy(test_case: LLMTestCase):
    metric = ContextualRelevancyMetric(
        threshold=0.7, model=azure_openai, include_reason=True
    )
    return d_evaluate([test_case], [metric])


def deepeval_evaluate_hallucination(item: EvaluationPayload):
    metric = HallucinationMetric(threshold=0.7, model=azure_openai, include_reason=True)
    test_case = LLMTestCase(
        input=item.prompt,
        actual_output=item.output,
        retrieval_context=item.documents,
        expected_output=item.expected_output,
        context=item.documents,
    )
    return d_evaluate([test_case], [metric])


def deepeval_g_eval(test_case: LLMTestCase):
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output. Note, that the retrieval context can be used to verify the correctness of the output.",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are OK",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=azure_openai,
        threshold=0.7,
    )

    return d_evaluate([test_case], [correctness_metric])


# ref: https://github.com/explodinggradients/ragas
# ref: https://docs.ragas.io/en/stable/concepts/metrics/index.html
def ragas_evaluate(item: EvaluationPayload):
    data_samples = {
        "question": [item.prompt],
        "answer": [item.output],
        "contexts": [[",".join(item.documents)]],
        "ground_truth": [item.expected_output],
    }

    dataset = Dataset.from_dict(data_samples)

    score = r_evaluate(
        dataset,
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
    df = score.to_pandas()
    # ref: https://stackoverflow.com/questions/71203579/how-to-return-a-csv-file-pandas-dataframe-in-json-format-using-fastapi
    return df.to_json(orient="records")
