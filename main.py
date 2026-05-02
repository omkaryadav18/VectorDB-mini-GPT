from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from document_db import DocumentDB
from chunker import chunk_text
from ollama_client import OllamaClient

from pydantic import BaseModel
from typing import List

from vector_db import VectorDB


app = FastAPI()

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

db = VectorDB(4)


db.insert(
    "Python Programming",
    "cs",
    [0.9, 0.8, 0.1, 0.1]
)

db.insert(
    "Machine Learning",
    "ai",
    [0.8, 0.9, 0.2, 0.1]
)

db.insert(
    "Pizza",
    "food",
    [0.1, 0.1, 0.9, 0.8]
)


class SearchRequest(BaseModel):

    vector: List[float]

    k: int = 5

    metric: str = "cosine"

    algo: str = "hnsw"


@app.get("/")
def home():

    return {
        "message": "Vector DB running"
    }


@app.post("/search")
def search(
    req: SearchRequest
):

    results = db.search(
        req.vector,
        req.k,
        req.metric,
        req.algo
    )

    return {
        "results": results
    }

doc_db = DocumentDB()

ollama = OllamaClient()

class DocumentRequest(BaseModel):

    title: str

    text: str


@app.post("/doc/insert")
def insert_document(
    req: DocumentRequest
):

    chunks = chunk_text(
        req.text
    )

    inserted_ids = []

    for chunk in chunks:

        embedding = ollama.embed(
            chunk
        )

        doc_id = doc_db.insert(

            req.title,

            chunk,

            embedding
        )

        inserted_ids.append(
            doc_id
        )

    return {

        "inserted_chunks": len(chunks),

        "ids": inserted_ids
    }

class AskRequest(BaseModel):

    question: str

    k: int = 3


@app.post("/ask")
def ask_ai(
    req: AskRequest
):

    q_embedding = ollama.embed(
        req.question
    )

    contexts = doc_db.search(

        q_embedding,

        req.k
    )

    context_text = ""

    for i, ctx in enumerate(
        contexts
    ):

        context_text += (
            f"[{i+1}] "
            + ctx["text"]
            + "\n\n"
        )

    prompt = f"""

Answer the user's question.

Context:
{context_text}

Question:
{req.question}

Answer:
"""

    answer = ollama.generate(
        prompt
    )

    return {

        "answer": answer,

        "contexts": contexts
    }

@app.get("/status")
def status():

    return {
        "ollamaAvailable": True,
        "embedModel": "nomic-embed-text",
        "genModel": "llama3.2",
        "docCount": len(doc_db.store)
    }