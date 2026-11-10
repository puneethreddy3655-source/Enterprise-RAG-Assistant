
import faiss
import numpy as np
import json

from pathlib import Path


class FAISSVectorStore:

    def __init__(self, embedding_dimension, storage_path="vector_store"):

        self.embedding_dimension = embedding_dimension

        self.storage_path = Path(storage_path)

        self.storage_path.mkdir(
            parents=True,
            exist_ok=True
        )

        # Exact similarity search using inner product
        self.index = faiss.IndexFlatIP(
            embedding_dimension
        )

        self.documents = []

    # Add embeddings and their associated documents
    def add_documents(self, embeddings, documents):

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix.")

        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError("Embedding dimension mismatch.")

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings and documents must match."
            )

        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embeddings contain invalid values.")

        # Add vectors to FAISS
        self.index.add(embeddings)

        # Preserve corresponding document records
        self.documents.extend(documents)

        print(
            f"Successfully indexed {len(documents)} chunks."
        )

    # Save FAISS index and document metadata
    def save(self):

        index_path = self.storage_path / "faiss.index"

        metadata_path = self.storage_path / "metadata.json"

        faiss.write_index(
            self.index,
            str(index_path)
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.documents,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("Vector index saved successfully!")

    # Load an existing FAISS index
    def load(self):

        index_path = self.storage_path / "faiss.index"

        metadata_path = self.storage_path / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "FAISS index or metadata file not found."
            )

        self.index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.documents = json.load(file)

        if self.index.d != self.embedding_dimension:
            raise ValueError("Stored index dimension mismatch.")

        if self.index.ntotal != len(self.documents):
            raise ValueError("Index and metadata are inconsistent.")

        print("Vector index loaded successfully!")

    # Search for similar vectors
    def search(self, query_embedding, top_k=3):

        if self.index.ntotal == 0:
            raise ValueError("Vector index is empty.")

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.embedding_dimension:
            raise ValueError("Query embedding dimension mismatch.")

        if not np.all(np.isfinite(query_embedding)):
            raise ValueError("Query embedding contains invalid values.")

        if not np.allclose(
            np.linalg.norm(query_embedding, axis=1),
            1.0,
            atol=1e-4
        ):
            raise ValueError("Query embedding must be normalized.")

        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0]
        ):

            if index_id == -1:
                continue

            results.append({
                "score": float(score),
                "document": self.documents[index_id]
            })

        return results
