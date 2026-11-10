
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingEngine:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):

        print("Loading embedding model...")

        self.model = SentenceTransformer(model_name)

        self.embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        print("Embedding model loaded successfully!")

        print(
            "Embedding dimensions:",
            self.embedding_dimension
        )

    # Generate embeddings for multiple text chunks
    def generate_embeddings(self, texts):

        if not texts:
            raise ValueError("No text provided for embedding.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings.astype(np.float32)

    # Generate embedding for one query
    def generate_query_embedding(self, query):

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.astype(np.float32)
