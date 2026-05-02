from brute_force import BruteForce
from kd_tree import KDTree
from hnsw import HNSW

from models import VectorItem
from distances import get_distance


class VectorDB:

    def __init__(self, dims):

        self.dims = dims

        self.store = {}

        self.next_id = 1

        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW()

    def insert(
        self,
        metadata,
        category,
        embedding
    ):

        item = VectorItem(
            self.next_id,
            metadata,
            category,
            embedding
        )

        self.next_id += 1

        self.store[item.id] = item

        self.bf.insert(item)
        self.kdt.insert(item)

        self.hnsw.insert(
            item,
            get_distance("cosine")
        )

        return item.id

    def search(
        self,
        query,
        k=5,
        metric="cosine",
        algo="hnsw"
    ):

        dist_fn = get_distance(
            metric
        )

        if algo == "bruteforce":

            results = self.bf.knn(
                query,
                k,
                dist_fn
            )

        elif algo == "kdtree":

            results = self.kdt.knn(
                query,
                k,
                dist_fn
            )

        else:

            results = self.hnsw.knn(
                query,
                k,
                dist_fn
            )

        output = []

        for dist, item_id in results:

            item = self.store[item_id]

            output.append({
                "id": item.id,
                "metadata": item.metadata,
                "category": item.category,
                "distance": float(dist)
            })

        return output