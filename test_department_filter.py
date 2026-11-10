
from src.retriever import Retriever


retriever = Retriever()

question = "What leave benefits are available to employees?"

results = retriever.retrieve(
    query=question,
    top_k=5,
    department="finance"
)

print("\nHR DEPARTMENT SEARCH")
print("=" * 60)

for result in results:

    document = result["document"]

    print("\nScore:", round(result["score"], 4))
    print("Department:", document["metadata"]["department"])
    print("Source:", document["metadata"]["source"])
    print("Content:", document["text"])
