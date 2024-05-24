import asyncio
from internal_shared.utils.timer import atime_wrapper
import tiktoken
from typing import Dict, List, Tuple
from fastapi import FastAPI, status
from uvicorn import Config, Server
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from llm import embed_text_async, invoke_prompt_async
from models import ChatRequest, ChatResponse, RetrievalConfig, RetrievalStep, TokenUsage
from retrieval import (
    PostRetrievalStrategyFactory,
    PostRetrievalType,
    PreRetrievalStrategyFactory,
    RetrievalStrategyFactory,
)


app = FastAPI()


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
async def chat(request: ChatRequest):
    # retrieve documents
    context, steps = await retrieve_documents(request)

    # prepare prompt
    prompt = prepare_prompt(context, request.query)

    # generate response
    res_time, response = await atime_wrapper(invoke_prompt_async, prompt)

    # calculate token usage
    token_usage = calculate_token_usage(prompt, response, request.model)

    return ChatResponse(
        response=response.content,
        documents=context,
        request=request.query,
        model=request.model,
        response_duration=res_time,
        token_usage=token_usage,
        steps=steps,
    )


def calculate_token_usage(
    prompt: List[BaseMessage], response: BaseMessage, model_name: str
):
    # check, if response.response_metadata has key token_usage
    if (
        hasattr(response, "response_metadata")
        and "token_usage" in response.response_metadata
    ):
        token_usage: Dict = response.response_metadata.get("token_usage")
        if token_usage and isinstance(token_usage, dict):
            print("Token usage present in response.response_metadata")

            completion_tokens = token_usage.get("completion_tokens", 0)
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            return TokenUsage(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=completion_tokens + prompt_tokens,
            )

    # case: token_usage not present in response.response_metadata
    prompt_string = "\n".join(p.content for p in prompt)
    encoded_prompt = calculate_token_length(prompt_string, model_name)
    encoded_response = calculate_token_length(response.content, model_name)

    return TokenUsage(
        completion_tokens=encoded_response,
        prompt_tokens=encoded_prompt,
        total_tokens=encoded_prompt + encoded_response,
    )


def calculate_token_length(prompt: str, model_name: str) -> int:
    """
    Calculate the token usage of a given prompt.

    Encoding list: https://github.com/openai/tiktoken/blob/c0ba74c238d18b4824c25f3c27fc8698055b9a76/tiktoken/model.py#L20
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    encoded_prompt = encoding.encode(prompt)
    return len(encoded_prompt)


async def retrieve_documents(
    request: ChatRequest,
) -> Tuple[List[str], List[RetrievalStep]]:
    async def retrieve(cfg: RetrievalConfig):
        # pre-retrieval
        pre = PreRetrievalStrategyFactory.create(cfg.pre_retrieval_type)
        pre_time, query = await atime_wrapper(pre.execute_async, request.query)

        embedded_query = await embed_text_async(query)

        # retrieval
        retrieval = RetrievalStrategyFactory.create(cfg.retrieval_type)
        ret_time, retrieved_documents = await atime_wrapper(
            retrieval.execute_async, embedded_query, cfg.threshhold, cfg.top_k
        )

        # post-retrieval
        post = PostRetrievalStrategyFactory.create(cfg.post_retrieval_type)
        post_time, post_retrieval_documents = await atime_wrapper(
            post.execute_async, retrieved_documents
        )

        # write times in RetrievalStep
        step = RetrievalStep(
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

        return context, step

    results = await asyncio.gather(
        *(retrieve(cfg) for cfg in request.retrieval_behaviour)
    )

    context: List[str] = []
    for c, _ in results:
        context.extend(c)
    steps = [step for _, step in results]

    return context, steps


def prepare_prompt(context: List[str], request: str):
    chat_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a highly efficient DevExpress Criteria Language Expression expert. Generate complex formulas swiftly and accurately based solely on the provided context and your DevExpress knowledge. Respond with the formula ONLY!",
            ),
            (
                "human",
                "Given the following context, please generate the appropriate DevExpress formula. Context:\n{context}\nYour expertise is required to formulate an accurate response quickly. Respond with the formula ONLY.",
            ),
            ("user", "{request}"),
        ]
    )
    return chat_template.format_messages(context=context, request=request)


def main():
    config = Config(app=app)
    server = Server(config=config)
    server.run()


if __name__ == "__main__":
    main()
