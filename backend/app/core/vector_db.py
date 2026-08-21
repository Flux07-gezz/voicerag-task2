import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

class FastVectorEngine:
    def __init__(self, collection_name="voice_rag_collection"):
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:")
        
        # 1. Explicitly load the FastEmbed model
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # 2. Manually create the vector collection (size 384 for BAAI model)
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            
    def index_chunks(self, chunks):
        if not chunks:
            return

        documents = [chunk.get("text", chunk.get("parent_context", "")) for chunk in chunks]
        
        # Manually generate embeddings
        embeddings = list(self.embedding_model.embed(documents))
        
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                payload=chunk
            )
            for vector, chunk in zip(embeddings, chunks)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_text: str, top_k: int = 2):
        # Embed the spoken question
        query_vector = list(self.embedding_model.embed([query_text]))[0]
        vector_list = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        
        try:
            # Use the new query_points method (replaces .search in newest Qdrant versions)
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector_list,
                limit=top_k
            )
            search_results = response.points
        except AttributeError:
            # Bulletproof fallback just in case
            search_results = self.client.query(
                collection_name=self.collection_name,
                query_vector=vector_list,
                limit=top_k
            )
        
        if not search_results:
            return [], 0.0
            
        # Extract the payload/metadata and the similarity score
        retrieved_contexts = [getattr(hit, "payload", getattr(hit, "metadata", {})) for hit in search_results]
        max_score = max([hit.score for hit in search_results]) if search_results else 0.0
        
        return retrieved_contexts, max_score
        # Embed the spoken question
        query_vector = list(self.embedding_model.embed([query_text]))[0]
        
        # Use native mathematical vector search (works on all versions)
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector),
            limit=top_k
        )
        
        if not search_results:
            return [], 0.0
            
        retrieved_contexts = [hit.payload for hit in search_results]
        max_score = max([hit.score for hit in search_results]) if search_results else 0.0
        
        return retrieved_contexts, max_score