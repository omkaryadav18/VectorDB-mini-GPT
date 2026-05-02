import heapq


class KDNode:

    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None


class KDTree:

    def __init__(self, dims):
        self.root = None
        self.dims = dims

    def insert(self, item):

        self.root = self._insert(
            self.root,
            item,
            depth=0
        )

    def _insert(self, node, item, depth):

        if node is None:
            return KDNode(item)

        axis = depth % self.dims

        if item.emb[axis] < node.item.emb[axis]:

            node.left = self._insert(
                node.left,
                item,
                depth + 1
            )

        else:

            node.right = self._insert(
                node.right,
                item,
                depth + 1
            )

        return node

    def knn(self, query, k, dist_fn):

        heap = []

        self._knn(
            self.root,
            query,
            k,
            0,
            dist_fn,
            heap
        )

        results = [
            (-d, item_id)
            for d, item_id in heap
        ]

        results.sort()

        return results

    def _knn(
        self,
        node,
        query,
        k,
        depth,
        dist_fn,
        heap
    ):

        if node is None:
            return

        dist = dist_fn(
            query,
            node.item.emb
        )

        if len(heap) < k:

            heapq.heappush(
                heap,
                (-dist, node.item.id)
            )

        elif dist < -heap[0][0]:

            heapq.heapreplace(
                heap,
                (-dist, node.item.id)
            )

        axis = depth % self.dims

        diff = (
            query[axis]
            - node.item.emb[axis]
        )

        close = (
            node.left
            if diff < 0
            else node.right
        )

        away = (
            node.right
            if diff < 0
            else node.left
        )

        self._knn(
            close,
            query,
            k,
            depth + 1,
            dist_fn,
            heap
        )

        if (
            len(heap) < k
            or abs(diff) < -heap[0][0]
        ):

            self._knn(
                away,
                query,
                k,
                depth + 1,
                dist_fn,
                heap
            )