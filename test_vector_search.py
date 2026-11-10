
from src.embeddings import EmbeddingEngine
from src.vector_store import FAISSVectorStore


# Step 1: Load embedding model

embedding_engine = EmbeddingEngine()


# Step 2: Initialize vector store

vector_store = FAISSVectorStore(
    embedding_dimension=384,
    storage_path="vector_store"
)


# Step 3: Load existing index

vector_store.load()


# Step 4: Ask a semantic question

query = "How many vacation days does an employee get?"


# Step 5: Convert query into embedding

query_embedding = embedding_engine.generate_query_embedding(
    query
)


# Step 6: Search similar document chunks

results = vector_store.search(
    query_embedding,
    top_k=3
)


# Step 7: Display results

print("\nUSER QUERY:", query)

print("\nRETRIEVED DOCUMENTS")

for i, result in enumerate(results):

    print("\n" + "=" * 60)

    print("Result:", i + 1)

    print("Similarity:", round(result["score"], 4))

    print("Source:",
          result["document"]["metadata"]["source"])

    print("Department:",
          result["document"]["metadata"]["department"])

    print("Text:")
    print(result["document"]["text"])
