from abc import ABC, abstractmethod
from enum import Enum
from typing import List
from retrieval.retrieval import SearchResult


class PostRetrievalType(str, Enum):
    DEFAULT = 'default'


class PostRetrievalStrategy(ABC):
    @abstractmethod
    def execute(self, documents: List[SearchResult]) -> List[SearchResult]:
        pass


class DefaultPostRetrievalStrategy(PostRetrievalStrategy):
    def execute(self, documents: List[SearchResult]) -> List[SearchResult]:
        return documents


class PostRetrievalStrategyFactory:
    @staticmethod
    def create(strategy_type: str) -> PostRetrievalStrategy:
        match strategy_type:
            case PostRetrievalType.DEFAULT:
                return DefaultPostRetrievalStrategy()
            case _:
                return DefaultPostRetrievalStrategy()
