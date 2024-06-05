import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.search.documents.models import VectorizedQuery
from neomodel import db, config
from internal_shared.models.chat import RetrievalConfig, SearchResult, RetrievalType

config.DATABASE_URL = os.getenv("NEO4J_URI")


class RetrievalStrategy(ABC):
    """
    Abstract class for retrieval strategies.
    """

    @abstractmethod
    def execute(
        self, query: List[float], threshold: float = 0.5, top_k: int = 5
    ) -> List[SearchResult]:
        """
        Execute a synchronous search query.

        :param query: A list of float values representing the query vector.
        :param threshold: The minimum score threshold for considering a result.
        :param top_k: The number of top results to retrieve.
        :return: A list of SearchResult objects.
        """
        pass

    @abstractmethod
    async def execute_async(
        self, query: List[float], threshold: float = 0.5, top_k: int = 5
    ) -> List[SearchResult]:
        """
        Execute an asynchronous search query.

        :param query: A list of float values representing the query vector.
        :param threshold: The minimum score threshold for considering a result.
        :param top_k: The number of top results to retrieve.
        :return: A list of SearchResult objects.
        """
        pass


class VectorDatabaseRetrievalStrategy(RetrievalStrategy):
    def __init__(self, index_name: str = None) -> None:
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
            index_name=index_name or os.getenv("AZURE_AI_SEARCH_INDEX"),
            credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY")),
        )

        self.async_search_client = AsyncSearchClient(
            endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
            index_name=index_name or os.getenv("AZURE_AI_SEARCH_INDEX"),
            credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY")),
        )

    def execute(
        self, query: List[float], threshold: float = 0.5, top_k: int = 5
    ) -> List[SearchResult]:
        vector_query = self._get_vectorized_query(query, top_k)
        results = self.search_client.search(
            vector_queries=[vector_query],
            select=["name", "summary", "content"],
        )
        documents = []
        for result in results:
            mapped_result = self._map_single_result(result, threshold)
            if mapped_result is not None:
                documents.append(mapped_result)
        return documents

    async def execute_async(
        self, query: List[float], threshold: float = 0.5, top_k: int = 5
    ) -> List[SearchResult]:
        """
        Making use of azure.search.documents.aio SearchClient, which relies on aiohttp.
        """
        vector_query = self._get_vectorized_query(query, top_k)
        async with self.async_search_client:
            results = await self.async_search_client.search(
                vector_queries=[vector_query],
                select=["name", "summary", "content"],
            )
            documents = []
            async for result in results:
                mapped_result = self._map_single_result(result, threshold)
                if mapped_result is not None:
                    documents.append(mapped_result)
            return documents

    def _get_vectorized_query(
        self, query: List[float], top_k: int, fields: str = "embedding"
    ) -> VectorizedQuery:
        return VectorizedQuery(
            vector=query,
            k_nearest_neighbors=top_k,
            fields=fields,
        )

    def _map_single_result(
        self, result: Dict, threshold: float
    ) -> Optional[SearchResult]:
        if "@search.score" not in result or "content" not in result:
            return None
        if result.get("@search.score", 0.0) < threshold:
            return None
        return SearchResult(
            name=result.get("name", "N/A"),
            summary=result.get("summary", ""),
            content=result.get("content", ""),
            score=float(result.get("@search.score", 0.0)),
            type=RetrievalType.VECTOR,
        )


class GraphDatabaseRetrievalStrategy(RetrievalStrategy):
    def execute(
        self, query: List[float], threshold: float = 0.5, top_k: int = 5
    ) -> List[SearchResult]:
        cypher_query = """  
        CALL db.index.vector.queryNodes('interface_embeddings', $num_neighbors, $query_embedding)  
        YIELD node AS similarNode, score  
        MATCH (similarNode)-[r:REFERENCES]->(relatedNode)  
        WHERE score >= $threshold  
        RETURN similarNode.name AS name, similarNode.summary AS summary, relatedNode.name AS related_name, score  
        """

        # Execute the query
        results, meta = db.cypher_query(
            cypher_query,
            {"num_neighbors": top_k, "query_embedding": query, "threshold": threshold},
        )

        # Process the results
        documents = []
        for result in results:
            documents.append(
                SearchResult(
                    name=result[0],
                    summary=result[1],
                    content=f"{result[0]} ({result[1]}) references {result[2]}",
                    score=result[3],
                    type=RetrievalType.GRAPH,
                )
            )

        return documents

    async def execute_async(
        self, query: List[float], threshold: float = 0.5, top_k: int = 5
    ) -> List[SearchResult]:
        """
        Execute an asynchronous search query by running the synchronous code in a separate thread.

        Since neomodel does not support async calls, we run the synchronous method in a separate thread.
        This should be safe to do, since reads are idempotent and do not modify the database.
        https://neomodel.readthedocs.io/en/latest/transactions.html#explicit-transactions
        """

        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor()

        # Run the synchronous method in a separate thread and await its completion
        return await loop.run_in_executor(
            executor, self.execute, query, threshold, top_k
        )


class RetrievalStrategyFactory:
    """
    Factory class to create retrieval strategies.
    """

    @staticmethod
    def create(cfg: RetrievalConfig) -> RetrievalStrategy:
        match cfg.retrieval_type:
            case RetrievalType.VECTOR:
                return VectorDatabaseRetrievalStrategy(index_name=cfg.index_name)
            case RetrievalType.GRAPH:
                return GraphDatabaseRetrievalStrategy()
            case _:
                raise ValueError(f"Unknown retrieval type: {cfg.retrieval_type}")


class RetrievalStep:
    """
    Facade class to execute retrieval strategies.
    """

    @staticmethod
    def execute(cfg: RetrievalConfig, query: List[float]) -> List[SearchResult]:
        """
        Execute a retrieval strategy based on the given configuration.
        """
        retrieval = RetrievalStrategyFactory.create(cfg)
        return retrieval.execute(query, cfg.threshold, cfg.top_k)

    @staticmethod
    async def execute_async(
        cfg: RetrievalConfig, query: List[float]
    ) -> List[SearchResult]:
        """
        Execute a retrieval strategy based on the given configuration asynchronously.
        """
        retrieval = RetrievalStrategyFactory.create(cfg)
        return await retrieval.execute_async(query, cfg.threshold, cfg.top_k)
