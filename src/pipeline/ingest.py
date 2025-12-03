"""Ingest chemical compound data into Qdrant with hybrid search (dense + sparse vectors)."""

import json
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv(override=True)

# Import data service
from ..services.data_service import get_data_service

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "chemistry_compounds")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen2.5-Embedding-0.6B")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "896"))

# Initialize clients
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
data_service = get_data_service()

def create_collection():
    """Create Qdrant collection with hybrid search configuration."""
    # Check if collection exists
    collections = qdrant_client.get_collections().collections
    collection_exists = any(c.name == COLLECTION_NAME for c in collections)

    if collection_exists:
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting...")
        qdrant_client.delete_collection(COLLECTION_NAME)

    # Create collection with dense + sparse vectors
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )
    print(f"Created collection '{COLLECTION_NAME}' with hybrid search enabled")

def load_compound_documents() -> List[Dict]:
    """Load all compound documents from TinyDB."""
    print("Loading compounds from TinyDB...")
    documents = data_service.get_all_compounds()
    print(f"Loaded {len(documents)} compound documents")
    return documents

def create_searchable_text(doc: Dict) -> str:
    """Create searchable text from all relevant fields."""
    return " ".join([
        doc.get("iupac_name", ""),
        " ".join(doc.get("common_names", [])),
        doc.get("formula", ""),
        doc.get("molecular_formula", ""),
        doc.get("class", ""),
        doc.get("info", ""),
        doc.get("naming_rule", "")
    ])

def ingest_documents(documents: List[Dict]):
    """Ingest documents into Qdrant with hybrid vectors using FastEmbed."""
    points = []

    for idx, doc in enumerate(documents):
        # Create searchable text from all relevant fields
        searchable_text = create_searchable_text(doc)

        # Create point with implicit embeddings using FastEmbed
        point = models.PointStruct(
            id=idx,
            vector={
                "dense": models.Document(
                    text=searchable_text,
                    model=EMBEDDING_MODEL_NAME
                ),
                "sparse": models.Document(
                    text=searchable_text,
                    model="Qdrant/bm25"
                )
            },
            payload=doc
        )
        points.append(point)

        print(f"Processed: {doc['doc_id']} ({doc['iupac_name']})")

    # Upload to Qdrant
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"Uploaded {len(points)} points to Qdrant")

def main():
    """Main ingestion workflow."""
    print("Starting ingestion process...")

    # Step 1: Create collection
    create_collection()

    # Step 2: Load documents
    documents = load_compound_documents()

    # Step 3: Ingest documents
    ingest_documents(documents)

    # Verify
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"\nCollection info:")
    print(f"  Points count: {collection_info.points_count}")
    print(f"  Vectors config: {collection_info.config.params.vectors}")
    print(f"  Sparse vectors config: {collection_info.config.params.sparse_vectors}")

    print("\nIngestion complete!")

if __name__ == "__main__":
    main()
