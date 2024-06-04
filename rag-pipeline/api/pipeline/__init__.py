import asyncio
import json
import time
from internal_shared.db.mongo import get_async_db
from internal_shared.models.chat import (
    ChatRequest,
    ChatResponse,
    ChatResponseChunk,
    RetrievalConfig,
    RetrievalStepResult,
    PostRetrievalType,
)
from internal_shared.utils.timer import atime_wrapper
from typing import Dict, List, Tuple
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessageChunk
from llm import embed_text_async, invoke_prompt_async, invoke_streaming_prompt_async
from retrieval import (
    PreRetrievalStep,
    RetrievalStep,
    PostRetrievalStep,
)
from .helper import calculate_token_str_usage, calculate_token_usage, format_chat_history


async def execute_pipeline(request: ChatRequest, chat_id: str):
    """
    Executes the whole pipeline for a given request.
    """
    # retrieve documents
    context, steps = await retrieve_documents(request)

    # prepare prompt
    prompt = await prepare_prompt(request, context)

    # generate response
    res_time, response = await atime_wrapper(invoke_prompt_async, prompt)

    # calculate token usage
    token_usage = calculate_token_usage(prompt, response, request.model)

    return ChatResponse(
        chat_session_id=chat_id,
        response=response.content,
        documents=context,
        request=request.query,
        model=request.model,
        response_duration=res_time,
        token_usage=token_usage,
        steps=steps,
    )


async def execute_pipeline_streaming(request: ChatRequest, chat_id: str, db_name: str):
    context, steps = await retrieve_documents(request)
    prompt = await prepare_prompt(request, context)

    start = time.perf_counter()
    chunks: List[BaseMessageChunk] = []
    async for chunk in invoke_streaming_prompt_async(prompt):
        chunks.append(chunk)
        yield json.dumps(
            ChatResponseChunk(chunk=chunk.content).model_dump(by_alias=True)
        )
    elapsed_time = (time.perf_counter() - start) * 1000

    response = "".join([c.content for c in chunks])

    # now, it is not important how long anything takes
    token_usage = calculate_token_str_usage(
        [p.content for p in prompt], response, request.model
    )

    # Send final response metadata as an empty chunk with complete metadata
    final_response_metadata = ChatResponse(
        chat_session_id=chat_id,
        response=response,
        documents=[item for sublist in context.values() for item in sublist],
        request=request.query,
        model=request.model,
        response_duration=elapsed_time,
        token_usage=token_usage,
        steps=steps,
    )

    # write to database
    db = await get_async_db(db_name)
    if db is not None:
        response_dto = final_response_metadata.to_dto_dict()
        await db.chat_response.insert_one(response_dto)

    yield json.dumps(
        ChatResponseChunk(
            chunk="", metadata=final_response_metadata.model_dump(by_alias=True)
        ).model_dump(by_alias=True)
    )


async def retrieve_documents(
    request: ChatRequest,
) -> Tuple[Dict[str, List[str]], List[RetrievalStepResult]]:
    """
    Executes the whole retrieval pipeline in order to retrieve documents based on the retrieval behaviour in the request.

    Args:
        - request (ChatRequest): The request object.
    """

    async def retrieve(cfg: RetrievalConfig):
        """
        Executes on retrieval step to retrieve documents based on the given retrieval configuration.
        """
        # pre-retrieval
        pre_time, query = await atime_wrapper(
            PreRetrievalStep.execute_async, cfg.pre_retrieval_type, request.query
        )

        embedded_query = await embed_text_async(query)

        # retrieval
        ret_time, retrieved_documents = await atime_wrapper(
            RetrievalStep.execute_async, cfg, embedded_query
        )

        # post-retrieval
        post_time, post_retrieval_documents = await atime_wrapper(
            PostRetrievalStep.execute_async, cfg, retrieved_documents
        )

        step = RetrievalStepResult(
            config=cfg,
            initial_query=request.query,
            pre_retrieval=query,
            pre_retrieval_duration=pre_time,
            retrieval=retrieved_documents,
            retrieval_duration=ret_time,
            post_retrieval=(
                post_retrieval_documents
                if cfg.post_retrieval_type != PostRetrievalType.DEFAULT
                else []
            ),
            post_retrieval_duration=post_time,
        )

        context = [doc.content for doc in post_retrieval_documents]

        return cfg.context_key, context, step

    results = await asyncio.gather(
        *(retrieve(cfg) for cfg in request.retrieval_behaviour)
    )

    context: Dict[str, List[str]] = {}
    for key, c, _ in results:
        if key in context:
            context[key].extend(c)
        else:
            context[key] = c
    steps = [step for _, _, step in results]

    return context, steps


async def prepare_prompt(request: ChatRequest, context: Dict[str, List[str]]):
    """
    Prepares the prompt for the chat API based on the context and the request.
    """

    context_strings = {key: "\n".join(value) for key, value in context.items()}

    chat_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant that helps people find information. Use the following context, to generate a response to the user request. Context:\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{request}"),
        ]
    )

    if request.prompt_template:
        chat_template = ChatPromptTemplate.from_messages(
            [
                ("system", request.prompt_template.template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{request}"),
            ]
        )

    combined_context = {
        **context_strings,
        "request": request.query,
        "chat_history": format_chat_history(request.history),
    }

    return await chat_template.aformat_messages(**combined_context)
