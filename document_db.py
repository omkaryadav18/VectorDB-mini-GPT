from models import VectorItem
from hnsw import HNSW
from distances import cosine


class DocumentDB:

    def __init__(self):

        self.store = {}

        self.next_id = 1

        self.hnsw = HNSW()

    def insert(
        self,
        title,
        text,
        embedding
    ):

        item = VectorItem(

            self.next_id,

            title,

            "document",

            embedding
        )

        self.store[
            item.id
        ] = {
            "title": title,
            "text": text,
            "embedding": embedding
        }

        self.hnsw.insert(
            item,
            cosine
        )

        self.next_id += 1

        return item.id

    def search(
        self,
        query_embedding,
        k=3
    ):

        results = self.hnsw.knn(

            query_embedding,

            k,

            cosine
        )

        output = []

        for dist, doc_id in results:

            doc = self.store[
                doc_id
            ]

            output.append({

                "id": doc_id,

                "title": doc["title"],

                "text": doc["text"],

                "distance": float(dist)
            })

        return output