from ollama_client import OllamaClient


ollama = OllamaClient()


embedding = ollama.embed(
    "What is machine learning?"
)

print(
    "Embedding dimensions:",
    len(embedding)
)


answer = ollama.generate(
    "Explain machine learning in simple words"
)

print(
    answer
)