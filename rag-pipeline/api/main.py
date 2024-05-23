import tiktoken
from typing import List, Tuple
from fastapi import FastAPI, status
from uvicorn import Config, Server
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from llm import invoke_prompt, embed_text
from retrieval.retrieval import RetrievalStrategyFactory, RetrievalType, SearchResult
from retrieval.pre_retrieval import PreRetrievalStrategyFactory, PreRetrievalType
from retrieval.post_retrieval import PostRetrievalStrategyFactory, PostRetrievalType
from internal_shared.ai_models.available_models import AvailableModels

_default_model = AvailableModels.GPT_4O


class RetrievalConfig(BaseModel):
    retrieval_type: RetrievalType
    pre_retrieval_type: PreRetrievalType
    post_retrieval_type: PostRetrievalType
    top_k: int = 5
    threshhold: float = 0.5


class RetrievalStep(BaseModel):
    config: RetrievalConfig
    query: str
    pre_retrieval: str
    retrieval: List[SearchResult]
    post_retrieval: List[SearchResult]


class ChatRequest(BaseModel):
    query: str
    retrieval_behaviour: List[RetrievalConfig]
    model: AvailableModels = _default_model


class ChatResponse(BaseModel):
    response: str
    request: str
    documents: List[str] | None = None
    model: AvailableModels = _default_model
    llm_duration: float = 0.0
    retrieval_duration: float = 0.0
    token_usage: float = 0.0
    steps: List[RetrievalStep] | None = None


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
def chat(request: ChatRequest):
    # retrieve documents
    context, steps = retrieve_documents(request)

    # prepare prompt
    prompt = prepare_prompt(context, request.query)

    # generate response
    response = invoke_prompt(prompt)

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


def retrieve_documents(
    request: ChatRequest,
) -> Tuple[List[str], List[RetrievalStep]]:
    context: List[str] = []
    steps: List[RetrievalStep] = []
    for cfg in request.retrieval_behaviour:
        # pre-retrieval: use different query strategies, e.g. None (traditional retrieval), query expansion, etc.
        pre = PreRetrievalStrategyFactory.create(cfg.pre_retrieval_type)
        query = pre.execute(request.query)

        # retrieval: could make request to Azure AI Search, neo4j (can make multiple requests to different sources!); use native libraries
        retrieval = RetrievalStrategyFactory.create(cfg.retrieval_type)
        embedded_query = embed_text(query)
        retrieved_documents = retrieval.execute(
            embedded_query, cfg.threshhold, cfg.top_k
        )

        # post-retrieval: use different post-processing strategies (e.g. re-ranking);
        post = PostRetrievalStrategyFactory.create(cfg.post_retrieval_type)
        post_retrieval_documents = post.execute(retrieved_documents)

        steps.append(
            RetrievalStep(
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
        )

        # ensure context is no list of lists
        [context.append(doc.search_result.content) for doc in post_retrieval_documents]

    return context, steps


def prepare_prompt(context: List[str], request: str):
    # current static prompt template
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
