import asyncio
from datetime import datetime, UTC
import json
from time import sleep
import time
from bson import ObjectId
from typing import AsyncGenerator, List, Optional
from fastapi import Query, FastAPI, Response, status
from fastapi.responses import StreamingResponse
from uvicorn import Config, Server
from internal_shared.db.mongo import get_async_db
from internal_shared.logger import get_logger
from internal_shared.models.chat import (
    ChatRequest,
    ChatResponse,
    ChatResponseChunk,
    PromptTemplate,
)
from pipeline import (
    execute_pipeline,
    execute_pipeline_streaming,
    prepare_prompt,
    retrieve_documents,
)
from uuid import uuid4
from llm import invoke_streaming_prompt_async

_RAG_PIPELINE_DB = "rag_pipeline"

app = FastAPI()

logger = get_logger(__name__)


@app.get("/")
def ping():
    return {"status": status.HTTP_200_OK}


@app.post(
    "/chat",
    summary="Chat with the AI",
    description="Chat with the AI to get the answer to your question",
    response_model=ChatResponse,
    response_model_by_alias=False,
)
async def chat(request: ChatRequest, chat_id: Optional[str] = None):
    cid = chat_id or str(uuid4())
    if chat_id:
        logger.info(f"Continuing chat with ID {chat_id}")
    else:
        logger.info(f"Starting new chat with ID {cid}")

    response = await execute_pipeline(request, cid)
    # write to DB
    db = await get_async_db(_RAG_PIPELINE_DB)
    if db is not None:
        response_dto = response.to_dto_dict()
        await db.chat_response.insert_one(response_dto)
    return response


@app.post(
    "/chat/stream",
    summary="Stream chat responses",
    description="Stream chat responses to the client",
    response_description="A stream of chat responses",
)
async def chat_stream(request: ChatRequest, chat_id: Optional[str] = None):
    cid = chat_id or str(uuid4())
    if chat_id:
        logger.info(f"Continuing chat with ID {chat_id}")
    else:
        logger.info(f"Starting new chat with ID {cid}")

    return StreamingResponse(
        execute_pipeline_streaming(request, cid, _RAG_PIPELINE_DB),
        media_type="text/event-stream",
    )


# endpoints to handle chat responses
@app.get("/chat_response")
async def get_response(
    from_date: datetime = Query(None),
    until_date: datetime = Query(None),
):
    query = {}
    date_query = {}
    if from_date:
        from_id = ObjectId.from_datetime(from_date)
        date_query["$gte"] = from_id
    if until_date:
        until_id = ObjectId.from_datetime(until_date)
        date_query["$lte"] = until_id
    else:
        until_id = ObjectId.from_datetime(datetime.now(UTC))
        date_query["$lte"] = until_id

    if date_query:
        query["_id"] = date_query
    return {"status": status.HTTP_200_OK}


# endpoints to handle prompt templates
@app.post(
    "/prompt_template",
    summary="Create a prompt template",
    description="Create a prompt template to be used for generating responses",
    response_description="The ID of the created template",
    response_model=str,
    response_model_by_alias=False,
)
async def create_template(template: PromptTemplate):
    db = await get_async_db(_RAG_PIPELINE_DB)
    if db is not None:
        template_dto = template.to_dto_dict()
        result = await db.prompt_template.insert_one(template_dto)
        return Response(
            status_code=status.HTTP_201_CREATED, content=str(result.inserted_id)
        )
    return Response(
        "Database not found", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )


@app.get(
    "/prompt_template",
    description="Get all prompt templates",
    response_description="A list of prompt templates",
    response_model=List[PromptTemplate],
    response_model_by_alias=False,
)
async def get_templates(skip: int = 0, limit: int = 100):
    db = await get_async_db(_RAG_PIPELINE_DB)
    if db is not None:
        templates = (
            await db.prompt_template.find({})
            .skip(skip)
            .limit(limit)
            .to_list(length=limit)
        )
        return templates
    return Response(
        content="Database not found", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )


@app.put(
    "/prompt_template/{template}",
    description="Update a prompt template",
    response_description="The ID of the updated template",
    response_model=str,
    response_model_by_alias=False,
)
async def update_template(template: str, new_template: PromptTemplate):
    db = await get_async_db(_RAG_PIPELINE_DB)
    if db is not None:
        if not ObjectId.is_valid(template):
            result = await db.prompt_template.update_one(
                {"_id": ObjectId(template)}, {"$set": new_template.to_dto_dict()}
            )
        else:
            result = await db.prompt_template.update_one(
                {"name": template}, {"$set": new_template.to_dto_dict()}
            )
        return Response(
            status_code=status.HTTP_200_OK, content=str(result.modified_count)
        )
    return Response(
        "Database not found", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )


@app.delete(
    "/prompt_template/{template}",
    description="Delete a prompt template",
    response_description="The ID of the deleted template",
    response_model=str,
    response_model_by_alias=False,
)
async def delete_template(template: str):
    # check if object id, else delete by name
    db = await get_async_db(_RAG_PIPELINE_DB)
    if db is not None:
        if ObjectId.is_valid(template):
            result = await db.prompt_template.delete_one({"_id": ObjectId(template)})
        else:
            result = await db.prompt_template.delete_one({"name": template})
        return Response(
            status_code=status.HTTP_200_OK, content=str(result.deleted_count)
        )


def main():
    config = Config(app=app)
    server = Server(config=config)
    server.run()


if __name__ == "__main__":
    main()
