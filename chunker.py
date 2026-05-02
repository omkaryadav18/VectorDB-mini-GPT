def chunk_text(
    text,
    chunk_size=250,
    overlap=30
):

    words = text.split()

    if len(words) <= chunk_size:

        return [text]

    chunks = []

    step = chunk_size - overlap

    for i in range(
        0,
        len(words),
        step
    ):

        chunk = words[
            i:i+chunk_size
        ]

        chunks.append(
            " ".join(chunk)
        )

        if i + chunk_size >= len(words):

            break

    return chunks