from abc import ABC, abstractmethod
from enum import Enum
from langchain_core.messages import HumanMessage, SystemMessage
from llm import invoke_prompt


class PreRetrievalType(str, Enum):
    DEFAULT = "default"
    QUERY_EXPANSION = "query_expansion"


class PreRetrievalStrategy(ABC):
    @abstractmethod
    def execute(self, query: str) -> str:
        pass


class DefaultPreRetrievalStrategy(PreRetrievalStrategy):
    """
    Default pre-retrieval strategy. No modifications to the query.
    """

    def execute(self, query: str) -> str:
        return query


class QueryExpansionPreRetrievalStrategy(PreRetrievalStrategy):
    """
    Query expansion strategy based on the paper "Query Expansion by Prompting Large Language Models" (https://arxiv.org/pdf/2305.03653)

    Compared to approaches like PRF, query expansion with LLMs is more flexible and can be used to expand queries with more complex semantics. This approach also relies on "general-purpose" LLMs, rather than specialized models for query expansion.
    """

    def execute(self, query: str) -> str:
        messages = [
            SystemMessage("Query expansion is required"),
            HumanMessage(
                f"Answer the following query: {query}\nGive the rationale before answering"
            ),
        ]
        response = invoke_prompt(messages)
        return (f"{query} " * 5) + response.content


class PreRetrievalStrategyFactory:
    """
    Factory class to create pre-retrieval strategies.
    """

    @staticmethod
    def create(strategy_type: PreRetrievalType) -> PreRetrievalStrategy:
        match strategy_type:
            case PreRetrievalType.DEFAULT:
                return DefaultPreRetrievalStrategy()
            case PreRetrievalType.QUERY_EXPANSION:
                return QueryExpansionPreRetrievalStrategy()
            case _:
                return DefaultPreRetrievalStrategy()
