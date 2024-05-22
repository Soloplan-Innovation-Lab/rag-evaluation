from pydantic import ConfigDict, BaseModel, Field
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from typing import Dict, List, Optional
from bson import ObjectId
from datetime import datetime

# ref: https://github.com/mongodb-developer/mongodb-with-fastapi/blob/master/app.py

# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.
PyObjectId = Annotated[str, BeforeValidator(str)]


# mapper model for deepeval metrics
class DeepEvalMetric(BaseModel):
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


# model for the evaluations collection
class Evaluations(BaseModel):
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
