from src.document_loader import DocumentLoader

# Initialize document loader
loader = DocumentLoader(
    data_directory="data/raw"
)

# Discover documents
files = loader.discover_documents()

print("Documents discovered:", len(files))

print("\nDocument names:")

for file in files:
    print(file.name)


# Load all documents
documents = loader.load_all_documents()

print("\nTotal extracted records:", len(documents))


# Display extracted information
for document in documents:

    print("\n" + "=" * 60)

    print("METADATA:")
    print(document["metadata"])

    print("\nEXTRACTED TEXT:")
    print(document["text"][:300])