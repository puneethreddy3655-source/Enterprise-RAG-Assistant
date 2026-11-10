from src.fixed_chunker import fixed_size_chunking

text = """
Employees are entitled to 24 days
of annual paid leave per year.

Employees are entitled to 12 days
of paid sick leave annually.

Employees can take 6 days of casual
leave per calendar year.
"""

chunks = fixed_size_chunking(
    text,
    chunk_size=100
)

for i, chunk in enumerate(chunks):

    print(f"\nCHUNK {i + 1}")
    print("-" * 40)
    print(chunk)