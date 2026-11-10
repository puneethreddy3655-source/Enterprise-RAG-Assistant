from src.document_loader import DocumentLoader
from src.text_splitter import RecursiveChunker


# Step 1: Load documents

loader = DocumentLoader("data/raw")

documents = loader.load_all_documents()

print("Documents loaded:", len(documents))


# Step 2: Initialize chunker

chunker = RecursiveChunker(
    chunk_size=250,
    chunk_overlap=40
)


# Step 3: Generate chunks

chunks = chunker.chunk_documents(documents)

print("\nTotal chunks generated:", len(chunks))


# Step 4: Display chunks

for chunk in chunks:

    print("\n" + "=" * 70)

    print("CHUNK ID:")
    print(chunk["metadata"]["chunk_id"])

    print("\nSOURCE:")
    print(chunk["metadata"]["source"])

    print("\nDEPARTMENT:")
    print(chunk["metadata"]["department"])

    print("\nCHUNK SIZE:")
    print(chunk["metadata"]["chunk_size"])

    print("\nCONTENT:")
    print(chunk["text"])


import json
from pathlib import Path

output_path = Path("data/processed/chunks.json")

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(output_path, "w", encoding="utf-8") as file:

    json.dump(
        chunks,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nChunks saved successfully!")
print("Location:", output_path)