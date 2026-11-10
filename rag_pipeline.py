
from src.retriever import Retriever
from src.gemini_generator import GeminiGenerator


class RAGPipeline:

    def __init__(self):

        print("Initializing RAG pipeline...")

        self.retriever = Retriever()
        self.generator = GeminiGenerator()

        print("RAG pipeline initialized successfully!")

    def ask(
        self,
        question,
        top_k=3,
        department=None,
        min_score=0.40
    ):

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            department=department,
            min_score=min_score
        )

        # Step 2: Handle empty retrieval
        if not retrieved_chunks:
            answer = (
                "Information is not available "
                "in the provided company documents."
            )

        else:
            # Step 3: Generate answer using retrieved context
            answer = self.generator.generate_answer(
                question=question,
                retrieved_chunks=retrieved_chunks
            )

        # Step 4: Prepare unique source documents
        sources = []
        seen_sources = set()

        for result in retrieved_chunks:

            document = result["document"]
            metadata = document["metadata"]

            source_name = metadata.get("source")

            # Keep only one entry per source file
            if source_name and source_name not in seen_sources:

                seen_sources.add(source_name)

                source_info = {
                    "source": source_name,
                    "department": metadata.get("department"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "similarity_score": round(
                        result["score"], 4
                    )
                }

                sources.append(source_info)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks
        }
