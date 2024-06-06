from datetime import datetime, UTC
import os
from bson import ObjectId
from typing import Optional
from fastapi import Query, FastAPI, status
from fastapi.responses import StreamingResponse
from uvicorn import Config, Server
from internal_shared.db.mongo import get_async_db
from internal_shared.logger import get_logger
from internal_shared.models.chat import (
    ChatRequest,
    ChatResponse,
)
from pipeline import execute_pipeline, execute_pipeline_streaming
from uuid import uuid4

from routers import retriever_config, prompt_template

_RAG_PIPELINE_DB = "rag_pipeline"

app = FastAPI()

app.include_router(retriever_config.router)
app.include_router(prompt_template.router)

logger = get_logger(__name__)


@app.get("/")
def ping():
    return {
        "status": status.HTTP_200_OK,
        "message": "Hello from the RAG pipeline API endpoint",
    }


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


def main():
    dev_port = os.getenv("RAG_DEV_PORT")
    if dev_port:
        config = Config(app=app, port=int(dev_port))
    else:
        config = Config(app=app)
    server = Server(config=config)
    server.run()


if __name__ == "__main__":
    main()
