from vector_db import VectorDB


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

query = [0.85, 0.85, 0.1, 0.1]

results = db.search(
    query,
    algo="hnsw"
)

print(results)