from typing import List
from bson import ObjectId
import bson
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from internal_shared.models.chat import PromptTemplate
from motor.motor_asyncio import AsyncIOMotorDatabase
from dependencies import PaginationParams, get_db

router = APIRouter(
    prefix="/prompt_template",
    tags=["prompt_template"],
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Database not found"}
    },
)


@router.get(
    "/",
    description="Get all prompt templates",
    response_description="A list of prompt templates",
    response_model=List[PromptTemplate],
    response_model_by_alias=False,
)
async def get_templates(
    filters: PaginationParams = Depends(), db: AsyncIOMotorDatabase = Depends(get_db)
):
    templates = (
        await db.prompt_template.find({})
        .skip(filters.skip)
        .limit(filters.limit)
        .to_list(length=filters.limit)
    )
    return templates


@router.post(
    "/",
    summary="Create a prompt template",
    description="Create a prompt template to be used for generating responses",
    response_description="The ID of the created template",
)
async def create_template(
    template: PromptTemplate, db: AsyncIOMotorDatabase = Depends(get_db)
):
    template_dto = template.to_dto_dict()
    result = await db.prompt_template.insert_one(template_dto)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED, content={"id": str(result.inserted_id)}
    )


@router.put(
    "/{template}",
    description="Update a prompt template",
    response_description="The ID of the updated template",
)
async def update_template(
    template: str,
    new_template: PromptTemplate,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if bson.is_valid(str.encode(template)):
        result = await db.prompt_template.update_one(
            {"_id": ObjectId(template)}, {"$set": new_template.to_dto_dict()}
        )
    else:
        result = await db.prompt_template.update_one(
            {"name": template}, {"$set": new_template.to_dto_dict()}
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"id": str(result.upserted_id)}
    )


@router.delete(
    "/{template}",
    description="Delete a prompt template",
    response_description="The ID of the deleted template",
)
async def delete_template(template: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    if bson.is_valid(str.encode(template)):
        result = await db.prompt_template.delete_one({"_id": ObjectId(template)})
    else:
        result = await db.prompt_template.delete_one({"name": template})
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"count": result.deleted_count}
    )
