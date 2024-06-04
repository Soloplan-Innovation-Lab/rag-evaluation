from abc import ABC, abstractmethod
from typing import List
from internal_shared.models.chat import (
    PostRetrievalType,
    RetrievalConfig,
    SearchResult,
)


class PostRetrievalStrategy(ABC):
    """
    Abstract class for post-retrieval strategies.
    """

    @abstractmethod
    def execute(self, documents: List[SearchResult]) -> List[SearchResult]:
        """
        Executes the post-retrieval strategy.
        """
        pass

    @abstractmethod
    async def execute_async(self, documents: List[SearchResult]) -> List[SearchResult]:
        """
        Executes the post-retrieval strategy asynchronously.
        """
        pass


class DefaultPostRetrievalStrategy(PostRetrievalStrategy):
    def execute(self, documents: List[SearchResult]) -> List[SearchResult]:
        return documents

    async def execute_async(self, documents: List[SearchResult]) -> List[SearchResult]:
        return documents


class PostRetrievalStrategyFactory:
    @staticmethod
    def create(strategy_type: str) -> PostRetrievalStrategy:
        match strategy_type:
            case PostRetrievalType.DEFAULT:
                return DefaultPostRetrievalStrategy()
            case _:
                return DefaultPostRetrievalStrategy()


class PostRetrievalStep:
    """
    Facade class to execute post-retrieval strategies.
    """

    @staticmethod
    def execute(cfg: RetrievalConfig, documents: List[SearchResult]) -> List[SearchResult]:
        """
        Executes a post-retrieval strategy based on the given configuration.
        """
        strat = PostRetrievalStrategyFactory.create(cfg.post_retrieval_type)
        return strat.execute(documents)

    @staticmethod
    async def execute_async(cfg: RetrievalConfig, documents: List[SearchResult]) -> List[SearchResult]:
        """
        Executes a post-retrieval strategy based on the given configuration asynchronously.
        """
        strat = PostRetrievalStrategyFactory.create(cfg.post_retrieval_type)
        return await strat.execute_async(documents)
