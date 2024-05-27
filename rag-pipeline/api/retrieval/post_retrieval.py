from abc import ABC, abstractmethod
from typing import List
from internal_shared.models.chat import PostRetrievalType, SearchResult


class PostRetrievalStrategy(ABC):
    @abstractmethod
    def execute(self, documents: List[SearchResult]) -> List[SearchResult]:
        pass

    @abstractmethod
    async def execute_async(self, documents: List[SearchResult]) -> List[SearchResult]:
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
