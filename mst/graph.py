import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        A = self.adj_mat
        n = A.shape[0]

        start = 0

        # MST adjacency matrix output
        mst = np.zeros_like(A, dtype=float)

        # Took help from chatgpt to understand every line of the implementation in the class slides
        # Conducted trial and error of my implementation with feedback from chatgpt 
        # Using approach from class slides using S, pi, pred
        in_mst = np.zeros(n, dtype=bool)
        pi = np.full(n, np.inf, dtype=float)   # pi[v] = best known edge weight to connect v to S
        pred = np.full(n, None)       # pred[v] = predecessor vertex in MST

        pi[start] = 0.0

        #  heapq stores (key=weight, vertex)
        heap = [(pi[v], v) for v in range(n)]
        heapq.heapify(heap)

        while heap:
            key_u, u = heapq.heappop(heap)

            # If u in mst already, skip
            if in_mst[u]:
                continue

            # Skip stale entries (old key values) that have been replaced by smaller discovered edges
            if key_u != pi[u]:
                continue

            # if u not in mst yet, add u to visited
            in_mst[u] = True

            # Add the edge (pred[u], u) to the MST (skip the start node which has no pred)
            if pred[u] != None:
                p = pred[u]
                w = A[p, u]
                # set both symmetric entries in mst matrix
                mst[p, u] = w
                mst[u, p] = w

            # edges out of u: update pi[v] for vertices not in MST
            for v in range(n):
                if in_mst[v] or v == u: # if vertex v is already in mst or is the same as the previous node, skip
                    continue
        
                # if vertex v is not in mst and is not u, get edge weight
                w = A[u, v]

                # if the weight is 0, it's not an edge
                if w == 0:
                    continue

                #if weight is less than current best edge weight connecting to v
                if w < pi[v]:
                    pi[v] = float(w) # set the best edge weight to this weight
                    pred[v] = u # set the new mst predecessor of v to u
                    heapq.heappush(heap, (pi[v], v))  # decrease the pi of this vertex to this new weight

        # handling detection of disconnected graph (MST doesn't span all vertices)
        # If some vertex never got a predecessor (except start), it's disconnected.
        if np.any((pred == -1) & (np.arange(n) != start)):
            raise ValueError("Graph is disconnected: MST does not span all vertices.")

        # store result
        self.mst = mst