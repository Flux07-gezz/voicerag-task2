"""
In-memory Qdrant Vector Engine with ONNX Quantized Embeddings.
Designed for sub-20ms hybrid context retrieval.
"""

from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding


class FastVectorEngine:
    def __init__(self, collection_name: str = "msmarco_chunks"):
        self.collection_name = collection_name
        # In-memory mode eliminates network I/O overhead
        self.client = QdrantClient(":memory:")

        # Local quantized ONNX embedding model (runs fast on CPU/GPU)
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.vector_dim = 384

        self._init_collection()

    def _init_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
        )

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Generates vectors and inserts point structs into Qdrant in-memory."""
        texts = [c["text"] for c in chunks]
        embeddings = list(self.embed_model.embed(texts))

        points = [
            PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    "text": chunk["text"],
                    "parent_context": chunk["parent_context"],
                    "metadata": chunk["metadata"],
                },
            )
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 3) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes dense vector search and returns context with top similarity score.
        """
        query_vector = list(self.embed_model.embed([query]))[0].tolist()

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        if not search_results:
            return [], 0.0

        max_score = search_results[0].score
        retrieved_contexts = [
            {
                "score": res.score,
                "child_text": res.payload["text"],
                "parent_context": res.payload["parent_context"],
                "metadata": res.payload["metadata"],
            }
            for res in search_results
        ]

        return retrieved_contexts, max_score