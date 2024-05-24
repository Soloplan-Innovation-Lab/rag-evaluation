from abc import ABC, abstractmethod
from enum import Enum
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from llm import invoke_prompt, invoke_prompt_async


class PreRetrievalType(str, Enum):
    DEFAULT = "default"
    QUERY_EXPANSION = "query_expansion"


class PreRetrievalStrategy(ABC):
    @abstractmethod
    def execute(self, query: str) -> str:
        pass

    @abstractmethod
    async def execute_async(self, query: str) -> str:
        pass


class DefaultPreRetrievalStrategy(PreRetrievalStrategy):
    """
    Default pre-retrieval strategy. No modifications to the query.
    """

    def execute(self, query: str) -> str:
        return query

    async def execute_async(self, query: str) -> str:
        return query


class QueryExpansionPreRetrievalStrategy(PreRetrievalStrategy):
    """
    Query expansion strategy based on the paper "Query Expansion by Prompting Large Language Models" (https://arxiv.org/pdf/2305.03653)

    Compared to approaches like PRF, query expansion with LLMs is more flexible and can be used to expand queries with more complex semantics. This approach also relies on "general-purpose" LLMs, rather than specialized models for query expansion.
    """

    def _get_messages(self, query: str) -> List:
        return [
            SystemMessage("Query expansion is required"),
            HumanMessage(
                f"Answer the following query: {query}\nGive the rationale before answering"
            ),
        ]

    def _format_query(self, query: str, response: str) -> str:
        return (f"{query} " * 5) + response

    def execute(self, query: str) -> str:
        messages = self._get_messages(query)
        response = invoke_prompt(messages)
        return (f"{query} " * 5) + response.content

    async def execute_async(self, query: str) -> str:
        messages = self._get_messages(query)
        response = await invoke_prompt_async(messages)
        return self._format_query(query, response.content)


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
