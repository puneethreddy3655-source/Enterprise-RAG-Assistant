
from src.retriever import Retriever


# Initialize the retrieval engine
retriever = Retriever()


# Employee questions
questions = [
    "How many annual paid leave days are available?",
    "How can I reset my company password?",
    "What is the daily meal allowance for business travel?",
    "When should I submit travel reimbursement claims?",
    "When can an employee get a replacement laptop?"
]


for question in questions:

    print("\n" + "=" * 80)
    print("EMPLOYEE QUESTION:", question)
    print("=" * 80)

    results = retriever.retrieve(
        query=question,
        top_k=3
    )

    if not results:
        print("No relevant information found.")
        continue

    for rank, result in enumerate(results, start=1):

        document = result["document"]
        metadata = document["metadata"]

        print(f"\nResult {rank}")
        print("-" * 40)

        print("Similarity Score:", round(result["score"], 4))
        print("Department:", metadata.get("department"))
        print("Source:", metadata.get("source"))
        print("Chunk ID:", metadata.get("chunk_id"))
        print("Content:", document["text"])
