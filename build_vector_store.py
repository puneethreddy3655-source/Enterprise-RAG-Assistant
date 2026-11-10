
import json

from src.embeddings import EmbeddingEngine
from src.vector_store import FAISSVectorStore


# STEP 1: Load processed chunks

with open(
    "data/processed/chunks.json",
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)

print("Total chunks loaded:", len(chunks))


# STEP 2: Extract text

texts = [
    chunk["text"]
    for chunk in chunks
]


# STEP 3: Initialize embedding model

embedding_engine = EmbeddingEngine()


# STEP 4: Generate embeddings

embeddings = embedding_engine.generate_embeddings(
    texts
)

print("\nEmbedding matrix shape:", embeddings.shape)


# STEP 5: Initialize FAISS vector store

vector_store = FAISSVectorStore(
    embedding_dimension=embedding_engine.embedding_dimension,
    storage_path="vector_store"
)


# STEP 6: Add vectors and metadata

vector_store.add_documents(
    embeddings=embeddings,
    documents=chunks
)


# STEP 7: Persist the vector database

vector_store.save()


print("\nEnterprise RAG indexing completed!")
