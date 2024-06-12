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
from internal_shared.models.chat import (
    RetrievalConfig,
    RetrieverConfig,
    SearchResult,
    RetrieverType,
)

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
    def __init__(self, retriever: RetrieverConfig) -> None:
        self.retriver = retriever

        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
            index_name=self.retriver.index_name,
            credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY")),
        )

        self.async_search_client = AsyncSearchClient(
            endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
            index_name=self.retriver.index_name,
            credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY")),
        )

    def execute(
        self, query: List[float], threshold: float = 0.5, top_k: int = 5
    ) -> List[SearchResult]:
        vector_query = self._get_vectorized_query(query, top_k)
        results = self.search_client.search(
            vector_queries=[vector_query],
            select=self.retriver.retriever_select,
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
                select=self.retriver.retriever_select,
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
        if "@search.score" not in result:
            return None

        mapped_results = {}

        for field, mapping in self.retriver.field_mappings.items():
            value = self._get_nested_value(result, mapping) or result.get(mapping, "")
            mapped_results[field] = value

        if result.get("@search.score", 0.0) < threshold:
            return None

        return SearchResult(
            name=mapped_results.get("name", "N/A"),
            summary=mapped_results.get("summary", ""),
            content=mapped_results.get("content", ""),
            score=float(result.get("@search.score", 0.0)),
            type=RetrieverType.VECTOR,
        )

    def _get_nested_value(self, d: Dict, keys: str) -> Optional[str]:
        keys_list = keys.split(".")
        for key in keys_list:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return None
        return d


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
                    type=RetrieverType.GRAPH,
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
        match cfg.retriever.retriever_type:
            case RetrieverType.VECTOR:
                return VectorDatabaseRetrievalStrategy(cfg.retriever)
            case RetrieverType.GRAPH:
                return GraphDatabaseRetrievalStrategy()
            case _:
                raise ValueError(f"Unknown retrieval type: {cfg}")


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
    
    @staticmethod
    def get_curated_documents() -> List[str]:
        """
        Get curated documents.

        #### Quick Note

        Current basic "workaround" implementation for curated formula context. This is used, because sometimes the request has no similarity with "required" documents. Thus, an analysis lead to these curated documents.

        For more "consistency", a check can be added, if a particular document is already in the retrieved documents. If so, this document can be skipped, as it is already present. Duplicates are not desired.
        """
        return [
            "FunctionName: [IsNullOrEmpty(String)]\nDescription: [Returns True if the specified String object is NULL or an empty string; otherwise, False is returned.]\nExample: [IsNullOrEmpty([ProductName])]",
            "FunctionName: [ToStr(Value)]\nDescription: [Returns a string representation of a specified value or property.]\nExample: [ToStr([ID])]",
            "FunctionName: [ItemsToText]\nDescription: [ItemsToText(CollectionPropertyPath, string SelectorExpression, [optional]string ItemToStringExpression, [optional]string Separator, [optional]string OrderExpression, [optional]bool distinct, [optional]bool IgnoreEmptyValues) Creates a textual representation of elements in a list. CollectionPropertyPath = path of the table SelectorExpression = used to select elements from the table ([*], [First], [Last], [n]) ItemToStringExpression = indicates the columns to be displayed, e.g. [Number] Separator = separator, e.g. , (comma) OrderExpression = indicates the column by which sorting is to take place (! = reverse sequence) Distinct = from now on, only unique values are returned. IgnoreEmptyValues = empty values are not displayed.]",
            "FunctionName: [ItemCount]\nDescription: [ItemCount(CollectionPropertyPath, string SelectorExpression, [optional]bool Distinct, [optional]bool DistinctExpression, [optional]bool IgnoreEmptyValues) Provides the number of list elements. CollectionPropertyPath = Path of the table SelectorExpression = Is used to select elements from the table ([*], [First], [Last], [n]) Distinct = Only unique values are output. DistinctExpression = Columns to be used for the uniqueness check. IgnoreEmptyValues = Empty values are not displayed.]",
            "FunctionName: [ToStrDate]\nDescription: [ToStrDate(DateTime,FormatString) Example: ToStrDate(Now(), 'yyyyMMdd') => 20191212 yy = year two-digit yyyy = year four-digit M = month in year MM = month two-digit dd = day two-digit hh = hour two-digit mm = minute two-digit ss = second of minute]",
            "FunctionName: [StrToDateTime]\nDescription: [StrToDateTime(string, [optional]string formatString) Returns a date-time value that is read from a character string. Via the formatString function, you can specify the date-time format or the language as an English name or as a name in the current CarLo® language. Example: StrToDateTime('22.02.2021 11:23', 'dd.MM.yyyy HH.mm ') or StrToDateTime ('05/15/2021 11:23 pm', 'English') yy = year, two digits yyyy = year, four digits M = month in the year MM =month, two digits dd = day, two digits HH = hour, two digits mm = minute, two digits ss = second of the minute ./ = separator]"
        ]
