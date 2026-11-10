
from src.embeddings import EmbeddingEngine
from src.vector_store import FAISSVectorStore


class Retriever:

    def __init__(self, storage_path="vector_store"):
        """
        Initialize the embedding engine and load
        the existing FAISS vector index.
        """

        self.embedding_engine = EmbeddingEngine()

        self.vector_store = FAISSVectorStore(
            embedding_dimension=self.embedding_engine.embedding_dimension,
            storage_path=storage_path
        )

        self.vector_store.load()

    def retrieve(
        self,
        query,
        top_k=3,
        department=None,
        min_score=None
    ):
        """
        Retrieve relevant document chunks for a query.

        Parameters:
            query: Employee's question
            top_k: Number of results to return
            department: Optional department filter
            min_score: Optional minimum similarity score

        Returns:
            List of retrieved document chunks with scores
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if min_score is not None:
            if not isinstance(min_score, (int, float)) or not -1 <= min_score <= 1:
                raise ValueError("min_score must be between -1 and 1.")

        # Step 1: Convert the question into an embedding
        query_embedding = self.embedding_engine.generate_query_embedding(
            query
        )

        # Step 2: Retrieve similarity scores for all indexed chunks.
        # Searching all chunks ensures department filtering does not
        # accidentally exclude a relevant result.
        total_chunks = self.vector_store.index.ntotal

        if total_chunks == 0:
            return []

        results = self.vector_store.search(
            query_embedding,
            top_k=total_chunks
        )

        # Step 3: Apply department filtering
        if department:
            results = [
                result for result in results
                if result["document"]["metadata"].get("department", "").lower()
                == department.lower()
            ]

        # Step 4: Apply minimum similarity threshold
        if min_score is not None:
            results = [
                result for result in results
                if result["score"] >= min_score
            ]

        # Step 5: Return only the requested number of results
        return results[:top_k]
