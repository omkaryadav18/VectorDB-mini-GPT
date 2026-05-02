import math
import random
import heapq


class HNSWNode:

    def __init__(self, item, max_layer):

        self.item = item
        self.max_layer = max_layer

        self.neighbors = [
            []
            for _ in range(
                max_layer + 1
            )
        ]


class HNSW:

    def __init__(
        self,
        M=16,
        ef_build=200
    ):

        self.graph = {}

        self.M = M
        self.M0 = M * 2

        self.ef_build = ef_build

        self.entry_point = None
        self.top_layer = -1

        random.seed(42)

    def random_level(self):

        return int(
            -math.log(
                random.random()
            )
        )

    def insert(
        self,
        item,
        dist_fn
    ):

        item_id = item.id

        level = self.random_level()

        node = HNSWNode(
            item,
            level
        )

        self.graph[item_id] = node

        if self.entry_point is None:

            self.entry_point = item_id
            self.top_layer = level

            return

        current = self.entry_point

        for layer in range(
            min(level, self.top_layer),
            -1,
            -1
        ):

            candidates = self.search_layer(
                item.emb,
                current,
                self.ef_build,
                layer,
                dist_fn
            )

            neighbors = [
                x[1]
                for x in candidates[:self.M]
            ]

            node.neighbors[layer] = neighbors

            for n_id in neighbors:

                other = self.graph[n_id]

                while len(
                    other.neighbors
                ) <= layer:

                    other.neighbors.append([])

                other.neighbors[layer].append(
                    item_id
                )

        if level > self.top_layer:

            self.top_layer = level
            self.entry_point = item_id

    def search_layer(
        self,
        query,
        entry_id,
        ef,
        layer,
        dist_fn
    ):

        visited = set()

        candidates = []
        found = []

        entry = self.graph[entry_id]

        dist = dist_fn(
            query,
            entry.item.emb
        )

        heapq.heappush(
            candidates,
            (dist, entry_id)
        )

        heapq.heappush(
            found,
            (-dist, entry_id)
        )

        visited.add(entry_id)

        while candidates:

            current_dist, current_id = heapq.heappop(
                candidates
            )

            current_node = self.graph[
                current_id
            ]

            if layer >= len(
                current_node.neighbors
            ):
                continue

            for neighbor_id in current_node.neighbors[layer]:

                if neighbor_id in visited:
                    continue

                visited.add(
                    neighbor_id
                )

                neighbor = self.graph[
                    neighbor_id
                ]

                d = dist_fn(
                    query,
                    neighbor.item.emb
                )

                if len(found) < ef:

                    heapq.heappush(
                        found,
                        (-d, neighbor_id)
                    )

                    heapq.heappush(
                        candidates,
                        (d, neighbor_id)
                    )

                elif d < -found[0][0]:

                    heapq.heapreplace(
                        found,
                        (-d, neighbor_id)
                    )

                    heapq.heappush(
                        candidates,
                        (d, neighbor_id)
                    )

        results = [
            (-d, item_id)
            for d, item_id in found
        ]

        results.sort()

        return results

    def knn(
        self,
        query,
        k,
        dist_fn
    ):

        if self.entry_point is None:
            return []

        results = self.search_layer(
            query,
            self.entry_point,
            max(k, 50),
            0,
            dist_fn
        )

        return results[:k]