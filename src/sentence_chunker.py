import re


def sentence_chunking(text, chunk_size=500):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 1 <= chunk_size:

            if current_chunk:
                current_chunk += " "

            current_chunk += sentence

        else:

            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks