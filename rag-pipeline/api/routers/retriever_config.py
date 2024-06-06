from typing import List
from bson import ObjectId
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from internal_shared.models.chat import RetrieverConfig
from motor.motor_asyncio import AsyncIOMotorDatabase
from dependencies import PaginationParams, get_db

router = APIRouter(
    prefix="/retriever_config",
    tags=["retriever_config"],
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Database not found"}
    },
)


@router.get(
    "/",
    description="Get all retriever configurations",
    response_description="A list of retriever configurations",
    response_model=List[RetrieverConfig],
    response_model_by_alias=False,
)
async def get_retrievers(
    filters: PaginationParams = Depends(), db: AsyncIOMotorDatabase = Depends(get_db)
):
    retriever_configs = (
        await db.retriever_config.find({})
        .skip(filters.skip)
        .limit(filters.limit)
        .to_list(length=filters.limit)
    )
    return retriever_configs


@router.post(
    "/",
    summary="Create a retriever configuration",
    description="Create a retriever configuration to be used for generating responses",
    response_description="The ID of the created configuration",
    response_model=str,
    response_model_by_alias=False,
)
async def create_retriever(
    config: RetrieverConfig, db: AsyncIOMotorDatabase = Depends(get_db)
):
    config_dto = config.to_dto_dict()
    result = await db.retriever_config.insert_one(config_dto)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED, content={"id": str(result.inserted_id)}
    )


@router.put(
    "/{config_id}",
    description="Update a retriever configuration",
    response_description="The ID of the updated configuration",
    response_model=str,
    response_model_by_alias=False,
)
async def update_retriever(
    config_id: str,
    new_config: RetrieverConfig,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    result = await db.retriever_config.update_one(
        {"_id": ObjectId(config_id)}, {"$set": new_config.to_dto_dict()}
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"id": str(result.upserted_id)}
    )


@router.delete(
    "/{config_id}",
    description="Delete a retriever configuration",
    response_description="The ID of the deleted configuration",
    response_model=str,
    response_model_by_alias=False,
)
async def delete_retriever(config_id: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    if ObjectId.is_valid(config_id):
        result = await db.retriever_config.delete_one({"_id": ObjectId(config_id)})
    else:
        result = await db.retriever_config.delete_one({"retriever_name": config_id})
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"count": result.deleted_count}
    )
