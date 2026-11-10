
import csv
import sys
from pathlib import Path

# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever


# Initialize retriever
retriever = Retriever()

# Load evaluation questions
csv_path = PROJECT_ROOT / "evaluation" / "test_questions.csv"

with open(csv_path, "r", encoding="utf-8") as file:
    test_cases = list(csv.DictReader(file))


top_k = 3

total_questions = len(test_cases)
successful_retrievals = 0

print("\nRETRIEVAL EVALUATION")
print("=" * 80)

for case in test_cases:

    question = case["question"]
    expected_department = case["expected_department"]
    expected_source = case["expected_source"]

    results = retriever.retrieve(
        query=question,
        top_k=top_k
    )

    retrieved_sources = [
        result["document"]["metadata"].get("source")
        for result in results
    ]

    retrieved_departments = [
        result["document"]["metadata"].get("department")
        for result in results
    ]

    hit = expected_source in retrieved_sources

    if hit:
        successful_retrievals += 1

    print("\nQuestion:", question)
    print("Expected source:", expected_source)
    print("Retrieved sources:", retrieved_sources)
    print("Expected department:", expected_department)
    print("Retrieved departments:", retrieved_departments)
    print("Result:", "PASS" if hit else "FAIL")


# Calculate Hit@K
hit_at_k = successful_retrievals / total_questions if total_questions else 0

print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

print("Total questions:", total_questions)
print("Successful retrievals:", successful_retrievals)
print(f"Hit@{top_k}: {hit_at_k:.2%}")
