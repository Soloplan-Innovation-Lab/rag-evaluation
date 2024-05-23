from typing_extensions import Annotated
from typing import Dict, List, Optional
from datetime import datetime

# try to import pydantic
try:
    from pydantic import ConfigDict, BaseModel, Field
    from pydantic.functional_validators import BeforeValidator
except ImportError:
    raise ImportError(
        "Necessary evaluation dependencies are not installed. Please run `pip install pydantic`."
    )

# try to import bson
try:
    from bson import ObjectId
except ImportError:
    raise ImportError(
        "Necessary evaluation dependencies are not installed. Please run `pip install pymongo`."
    )

# ref: https://github.com/mongodb-developer/mongodb-with-fastapi/blob/master/app.py

# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.
PyObjectId = Annotated[str, BeforeValidator(str)]


class DeepEvalMetric(BaseModel):
    """
    Mapper model for the deepeval metrics.

    Args:
        - reason (str | None): The reason for the metric. Defaults to None.
        - score (float): The score of the metric.
        - success (bool | None): The success of the metric. Defaults to None.
        - threshold (float): The threshold of the metric.
        - additional_information (str | None): Additional information for the metric. Defaults to None.
    """

    reason: Optional[str] = None
    score: float
    success: Optional[bool] = None
    threshold: float
    additional_information: Optional[str] = None

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )


# model for the runs collection
class Runs(BaseModel):
    """
    The model for the runs collection.

    Args:
        - run_type (str): The type of the run (e.g. formula, knowledge, workflow, etc.).
        - category (str | None): The category of the run. Defaults to None.
        - tags (List[str] | None): The tags of the run. Defaults to None.
        - description (str): The description of the run.
        - start_time (datetime): The start time of the run.
        - end_time (datetime | None): The end time of the run. Defaults to None.
        - total_data_points (int): The total data points of the run.
    """

    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    run_type: str
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    description: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_data_points: int

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )


class Evaluations(BaseModel):
    """
    The model for the evaluations collection.

    Args:
        - run_id (str): The run ID.
        - total_iterations (int): The total iterations of the evaluation.
        - input (str): The input to the model.
        - actual_output (str): The actual output from the model.
        - expected_output (str | None): The expected output from the model. Defaults to None.
        - context (List[str] | None): The context for the model. Defaults to None.
        - retrieval_context (List[str]): The retrieval context for the model.
    """

    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    run_id: PyObjectId
    total_iterations: int
    input: str
    actual_output: str
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: List[str]

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )


# model for the iterations collection
class Iterations(BaseModel):
    """
    The model for the iterations collection.

    Args:
        - evaluation_id (str): The evaluation ID.
        - iteration (int): The iteration number.
        - deepeval (Dict[str, DeepEvalMetric]): The deepeval metrics.
        - ragas (Dict[str, float]): The ragas metrics.
    """

    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    evaluation_id: PyObjectId
    iteration: int
    deepeval: Dict[str, DeepEvalMetric]
    ragas: Dict[str, float]

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )


# populated classes
class PopulatedEvaluations(Evaluations):
    iterations: List[Iterations]


class PopulatedRuns(Runs):
    evaluations: List[PopulatedEvaluations]


# internal types used to correctly save the data into the database
# this means, that the ObjectId is used instead of the string representation
class _Evaluations(Evaluations):
    run_id: ObjectId


class _Iterations(Iterations):
    evaluation_id: ObjectId
