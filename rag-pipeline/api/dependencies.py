from fastapi import HTTPException, status
from internal_shared.db.mongo import get_async_db
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field

RAG_PIPELINE_DB = "rag_pipeline"


class PaginationParams(BaseModel):
    skip: int = Field(0, ge=0)
    limit: int = Field(100, ge=1)


async def get_db() -> AsyncIOMotorDatabase:
    db = await get_async_db(RAG_PIPELINE_DB)
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not found",
        )
    return db
