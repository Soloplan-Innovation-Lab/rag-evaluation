from typing import List

try:
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "Necessary evaluation dependencies are not installed. Please run `pip install pydantic`."
    )


class EvaluationPayload(BaseModel):
    """
    The payload for each evaluation entry.

    Args:
        - input (str): The input to the model.
        - actual_output (str): The actual output from the model.
        - expected_output (str | None): The expected output from the model. Defaults to None.
        - context (List[str] | None): The context for the model. Defaults to None.
        - retrieval_context (List[str]): The retrieval context for the model.
    """

    input: str
    actual_output: str
    expected_output: str | None = None
    context: List[str] | None = None
    retrieval_context: List[str]


class EvaluationRequest(BaseModel):
    """
    The request for an evaluation.

    Args:
        - dataset (List[EvaluationPayload]): The dataset to evaluate.
        - iterations_per_entry (int): The amount of iterations for each document/payload entry.
        - description (str): The description of the evaluation.
        - run_type (str): The type of run (e.g. formula, knowledge, workflow, etc.).
    """

    dataset: List[EvaluationPayload]
    iterations_per_entry: int
    description: str
    run_type: str


class EvaluationResult(BaseModel):
    """
    The result of an evaluation.

    Args:
        - evaluation_batch_number (int): The evaluation batch number.
        - evaluation_id (str): The evaluation ID.
        - iteration_ids (List[str]): The iteration IDs.
    """

    evaluation_batch_number: int
    evaluation_id: str
    iteration_ids: List[str]


class EvaluationResponse(BaseModel):
    """
    The response for an evaluation.

    Args:
        - run_id (str): The run ID.
        - evaluation_ids (List[EvaluationResult]): The evaluation IDs.
    """

    run_id: str
    evaluation_ids: List[EvaluationResult]
