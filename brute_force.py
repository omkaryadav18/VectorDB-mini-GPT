class BruteForce:

    def __init__(self):
        self.items = []

    def insert(self, item):
        self.items.append(item)

    def remove(self, item_id):
        self.items = [
            x for x in self.items
            if x.id != item_id
        ]

    def knn(self, query, k, dist_fn):

        results = []

        for item in self.items:

            dist = dist_fn(query, item.emb)

            results.append(
                (dist, item.id)
            )

        results.sort(
            key=lambda x: x[0]
        )

        return results[:k]