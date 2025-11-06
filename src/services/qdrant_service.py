"""Qdrant client helper for hybrid search."""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv(override=True)


class QdrantService:
    """Service class for Qdrant hybrid search operations."""

    def __init__(self):
        """Initialize Qdrant client."""
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "chemistry_compounds")
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.top_k = int(os.getenv("RAG_TOP_K", "3"))
        self.score_threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.4"))

        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            check_compatibility=False
        )

    def hybrid_search(self, query: str, limit: Optional[int] = None, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search (dense + sparse) with RRF fusion.

        Args:
            query: Search query text
            limit: Number of results to return (default: from env RAG_TOP_K)
            threshold: Minimum score threshold (default: from env RAG_SCORE_THRESHOLD)

        Returns:
            List of document payloads from top-K results with score >= threshold
        """
        if limit is None:
            limit = self.top_k
        if threshold is None:
            threshold = self.score_threshold

        # Hybrid search with RRF fusion
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            prefetch=[
                # Dense vector search
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=self.embedding_model
                    ),
                    using="dense",
                    limit=10  # Prefetch more for better RRF
                ),
                # Sparse BM25 search
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="Qdrant/bm25"
                    ),
                    using="sparse",
                    limit=10
                )
            ],
            limit=limit,
            with_payload=True
        )

        # Extract payloads with scores, filter by threshold
        return [
            {**point.payload, "score": point.score}
            for point in results.points
            if point.score >= threshold
        ]


# Global instance
qdrant_service = QdrantService()
