import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import List
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from neomodel import db, config
from pydantic import BaseModel

config.DATABASE_URL = os.getenv("NEO4J_URI")

# check connection
try:
    # Attempt to run a simple query
    db.cypher_query("MATCH (n) RETURN count(n)")
    print("Neo4j connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")

search_client = SearchClient(
    endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_AI_SEARCH_INDEX"),
    credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY")),
)

# check connection
try:
    search_client.get_document_count()
    print("Search client connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")


class RetrievalType(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"


class SearchResultContent(BaseModel):
    name: str
    summary: str
    content: str


class SearchResult(BaseModel):
    search_result: SearchResultContent
    score: float
    type: RetrievalType


class RetrievalStrategy(ABC):
    @abstractmethod
    def execute(
        self, query: List[float], threshold: float = 0.5, top_k: int = 5
    ) -> List[SearchResult]:
        pass


class VectorDatabaseRetrievalStrategy(RetrievalStrategy):
    def execute(
        self, query: List[float], threshold: float = 0.5, top_k: int = 5
    ) -> List[SearchResult]:
        vector_query = VectorizedQuery(
            vector=query,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )
        results = search_client.search(
            vector_queries=[vector_query],
            select=["name", "summary", "content"],
        )

        documents: List[SearchResult] = []
        for result in results:
            # if score or content couldnt be found, skip
            if "@search.score" not in result or "content" not in result:
                continue
            if result.get("@search.score", 0.0) < threshold:
                continue
            documents.append(
                SearchResult(
                    search_result=SearchResultContent(
                        name=result.get("name", "N/A"),
                        summary=result.get("summary", ""),
                        content=result.get("content", ""),
                    ),
                    score=float(result.get("@search.score", 0.0)),
                    type=RetrievalType.VECTOR,
                )
            )

        return documents


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
                    search_result=SearchResultContent(
                        name=result[0],
                        summary=result[1],
                        content=f"{result[0]} ({result[1]}) references {result[2]}",
                    ),
                    score=result[3],
                    type=RetrievalType.GRAPH,
                )
            )

        return documents


class CombinedRetrievalStrategy(RetrievalStrategy):
    def execute(
        self, query: List[float], threshold: float = 0.5, top_k: int = 5
    ) -> List[SearchResult]:
        vector_documents = VectorDatabaseRetrievalStrategy().execute(
            query, threshold, top_k
        )
        graph_documents = GraphDatabaseRetrievalStrategy().execute(
            query, threshold, top_k
        )
        combined_documents = vector_documents + graph_documents
        return combined_documents


class RetrievalStrategyFactory:
    @staticmethod
    def create(retrieval_type: RetrievalType) -> RetrievalStrategy:
        match retrieval_type:
            case RetrievalType.VECTOR:
                return VectorDatabaseRetrievalStrategy()
            case RetrievalType.GRAPH:
                return GraphDatabaseRetrievalStrategy()
            case RetrievalType.COMBINED:
                return CombinedRetrievalStrategy()
            case _:
                raise ValueError(f"Unknown retrieval type: {retrieval_type}")
