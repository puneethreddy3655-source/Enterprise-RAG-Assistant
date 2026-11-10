import json

from src.embeddings import EmbeddingEngine


# Step 1: Load processed chunks

with open(
    "data/processed/chunks.json",
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


# Step 2: Extract text

texts = [
    chunk["text"]
    for chunk in chunks
]


# Step 3: Initialize embedding engine

embedding_engine = EmbeddingEngine()


# Step 4: Generate embeddings

embeddings = embedding_engine.generate_embeddings(texts)


# Step 5: Inspect output

print("\nNumber of chunks:", len(texts))

print("Embedding shape:", embeddings.shape)

print("Embedding dimension:", embeddings.shape[1])

print("\nFirst embedding:")

print(embeddings[0])


# Step 6: Check normalization

import numpy as np

print(
    "\nFirst vector norm:",
    np.linalg.norm(embeddings[0])
)