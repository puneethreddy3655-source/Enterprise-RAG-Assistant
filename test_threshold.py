
from src.retriever import Retriever


retriever = Retriever()

question = "What is the international relocation policy?"

results = retriever.retrieve(
    query=question,
    top_k=3,
    min_score=0.50
)

if not results:
    print("No sufficiently relevant information found.")
else:
    print("Relevant information found:")

    for result in results:
        print("\nScore:", round(result["score"], 4))
        print("Content:", result["document"]["text"])
