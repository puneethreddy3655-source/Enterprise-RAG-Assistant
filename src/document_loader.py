
from pathlib import Path
import pymupdf
from docx import Document


class DocumentLoader:

    def __init__(self, data_directory):

        self.data_directory = Path(data_directory)

        self.supported_extensions = [
            ".pdf",
            ".docx",
            ".txt"
        ]

    # Discover all supported documents
    def discover_documents(self):

        files = []

        for file_path in self.data_directory.rglob("*"):

            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in self.supported_extensions
            ):

                files.append(file_path)

        return files

    # Extract text from TXT files
    def load_txt(self, file_path):

        text = file_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        return [{
            "text": text,
            "page": None
        }]

    # Extract text from PDF files page by page
    def load_pdf(self, file_path):

        extracted_pages = []

        with pymupdf.open(file_path) as pdf:

            for page_number, page in enumerate(pdf):

                text = page.get_text("text")

                if text.strip():

                    extracted_pages.append({
                        "text": text,
                        "page": page_number + 1
                    })

        return extracted_pages

    # Extract text from DOCX files
    def load_docx(self, file_path):

        document = Document(file_path)

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        text = "\n".join(paragraphs)

        return [{
            "text": text,
            "page": None
        }]

    # Load an individual document
    def load_document(self, file_path):

        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return self.load_pdf(file_path)

        elif extension == ".docx":
            return self.load_docx(file_path)

        elif extension == ".txt":
            return self.load_txt(file_path)

        else:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

    # Load all documents and attach metadata
    def load_all_documents(self):

        all_documents = []

        files = self.discover_documents()

        for file_path in files:

            try:

                pages = self.load_document(file_path)

                department = file_path.parent.name

                for item in pages:

                    if not item["text"].strip():
                        continue

                    record = {
                        "text": item["text"],
                        "metadata": {
                            "source": file_path.name,
                            "file_path": str(file_path),
                            "department": department,
                            "file_type": file_path.suffix.lower(),
                            "page": item["page"]
                        }
                    }

                    all_documents.append(record)

                print(
                    f"Successfully loaded: {file_path.name}"
                )

            except Exception as error:

                print(
                    f"Failed to load {file_path.name}: {error}"
                )

        return all_documents
