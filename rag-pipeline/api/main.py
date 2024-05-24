import asyncio
from internal_shared.utils.timer import async_timer, timer
import tiktoken
from typing import List, Tuple
from fastapi import FastAPI, status
from uvicorn import Config, Server
from langchain.prompts import ChatPromptTemplate
from llm import embed_text_async, invoke_prompt_async
from models import ChatRequest, ChatResponse, RetrievalConfig, RetrievalStep
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


@async_timer
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
    response = await invoke_prompt_async(prompt)

    # measure prompt usage
    prompt_string = ""
    for p in prompt:
        prompt_string += p.content + " "
    encoding = tiktoken.get_encoding("cl100k_base")
    encoded_prompt = encoding.encode(prompt_string)

    return ChatResponse(
        response=response.content,
        request=request.query,
        documents=context,
        token_usage=len(encoded_prompt),
        steps=steps,
    )


@async_timer
async def retrieve_documents(
    request: ChatRequest,
) -> Tuple[List[str], List[RetrievalStep]]:
    @async_timer
    async def retrieve(cfg: RetrievalConfig):
        # pre-retrieval
        pre = PreRetrievalStrategyFactory.create(cfg.pre_retrieval_type)
        query = await pre.execute_async(request.query)
        embedded_query = await embed_text_async(query)

        # retrieval
        retrieval = RetrievalStrategyFactory.create(cfg.retrieval_type)
        retrieved_documents = await retrieval.execute_async(
            embedded_query, cfg.threshhold, cfg.top_k
        )

        # post-retrieval
        post = PostRetrievalStrategyFactory.create(cfg.post_retrieval_type)
        post_retrieval_documents = await post.execute_async(retrieved_documents)

        step = RetrievalStep(
            config=cfg,
            query=request.query,
            pre_retrieval=query,
            retrieval=retrieved_documents,
            post_retrieval=(
                post_retrieval_documents
                if cfg.post_retrieval_type != PostRetrievalType.DEFAULT
                else []
            ),
        )

        context = [doc.search_result.content for doc in post_retrieval_documents]

        return context, step

    results = await asyncio.gather(
        *(retrieve(cfg) for cfg in request.retrieval_behaviour)
    )

    context: List[str] = []
    for c, _ in results:
        context.extend(c)
    steps = [step for _, step in results]

    return context, steps


@timer
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
