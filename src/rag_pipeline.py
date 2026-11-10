
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

        # ---------------------------------
        # STEP 1: RETRIEVE RELEVANT CHUNKS
        # ---------------------------------

        retrieved_chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            department=department,
            min_score=min_score
        )

        # ---------------------------------
        # STEP 2: GENERATE ANSWER
        # ---------------------------------

        if retrieved_chunks:

            answer = self.generator.generate_answer(
                question=question,
                retrieved_chunks=retrieved_chunks
            )

        else:

            answer = (
                "Information is not available "
                "in the provided company documents."
            )

        # ---------------------------------
        # STEP 3: CHECK ANSWERABILITY
        # ---------------------------------

        unavailable_messages = [
            "information is not available",
            "could not find sufficient information",
            "not available in the provided company documents"
        ]

        answer_lower = answer.lower()

        answer_unavailable = any(
            message in answer_lower
            for message in unavailable_messages
        )

        # ---------------------------------
        # STEP 4: REMOVE DUPLICATE SOURCES
        # ---------------------------------

        sources_by_name = {}

        if not answer_unavailable:

            for result in retrieved_chunks:

                document = result["document"]
                metadata = document["metadata"]

                source_name = metadata.get("source")

                if not source_name:
                    continue

                # Keep only the first occurrence.
                # Retrieval results are ranked by similarity,
                # so the first occurrence has the highest score.
                if source_name not in sources_by_name:

                    sources_by_name[source_name] = {
                        "source": source_name,
                        "department": metadata.get("department"),
                        "page": metadata.get("page"),
                        "chunk_id": metadata.get("chunk_id"),
                        "similarity_score": round(
                            result["score"], 4
                        )
                    }

        sources = list(sources_by_name.values())

        # ---------------------------------
        # STEP 5: RETURN FINAL RESULT
        # ---------------------------------

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks
        }

