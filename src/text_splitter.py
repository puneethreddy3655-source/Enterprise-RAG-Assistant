
class RecursiveChunker:

    def __init__(self, chunk_size=500, chunk_overlap=80):

        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "Overlap must be non-negative and smaller than chunk size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.separators = [
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]

    # Recursively divide text using preferred separators
    def _recursive_split(self, text, separators):

        if len(text) <= self.chunk_size:
            return [text] if text else []

        if not separators:
            return [
                text[i:i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        separator = separators[0]

        if separator == "":
            return [
                text[i:i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        if separator not in text:
            return self._recursive_split(
                text,
                separators[1:]
            )

        parts = text.split(separator)

        units = []

        for i, part in enumerate(parts):

            # Preserve the original separator
            piece = part

            if i < len(parts) - 1:
                piece += separator

            if not piece:
                continue

            if len(piece) > self.chunk_size:

                units.extend(
                    self._recursive_split(
                        piece,
                        separators[1:]
                    )
                )

            else:
                units.append(piece)

        return units

    # Combine smaller units into final chunks
    def _merge_units(self, units):

        chunks = []
        current_chunk = ""

        for unit in units:

            if len(current_chunk) + len(unit) <= self.chunk_size:

                current_chunk += unit

            else:

                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Carry forward overlapping text
                overlap_text = current_chunk[-self.chunk_overlap:]

                # Ensure the next chunk doesn't exceed its size
                available_space = self.chunk_size - len(unit)

                overlap_text = overlap_text[-available_space:] if available_space > 0 else ""

                current_chunk = overlap_text + unit

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    # Main public method
    def split_text(self, text):

        if not text or not text.strip():
            return []

        units = self._recursive_split(
            text.strip(),
            self.separators
        )

        return self._merge_units(units)

    # Process complete document records
    def chunk_documents(self, documents):

        all_chunks = []

        for document in documents:

            text = document["text"]
            metadata = document["metadata"]

            chunks = self.split_text(text)

            for index, chunk in enumerate(chunks):

                chunk_record = {

                    "text": chunk,

                    "metadata": {
                        **metadata,
                        "chunk_index": index,
                        "chunk_id": (
                            f"{metadata['source']}_"
                            f"{metadata.get('page')}_"
                            f"{index}"
                        ),
                        "chunk_size": len(chunk)
                    }
                }

                all_chunks.append(chunk_record)

        return all_chunks
