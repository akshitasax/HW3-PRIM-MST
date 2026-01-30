import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances
from collections import deque


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_edges():
    """
    Checks if MST has the correct number of edges, that is, v-1    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    mst = g.construct_mst()
    n_v = mst.shape[0]

    mst_edges = 0
    for i in range(mst.shape[0]): #for each row in mst
        for j in range(i+1): #for each col in mst
            if mst[i,j] != 0: #if row,col is not 0, that is, this is an edge
                mst_edges += 1 #add to tally of edges

    assert mst_edges == n_v-1 #assert that tally of edges is same as computed number of vertices -1


def test_mst_connected():
    """
    Checks if MST is connected - all vertices are visited if we do a search (DFS)   
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    mst = g.construct_mst() #create MST
    n_v = mst.shape[0] #get number of vertices

    visited = set() #make empty set of visited vertices
    q = deque([0]) #create priority queue and add start node (0) to it
    visited.add(0)

    while q: #while the q is not empty
        u = q.popleft() #get earliest added vertex
        for v in range(n_v): #for each vertex in mst
            if mst[u, v] != 0 and v not in visited: #if the connection (v) to the earliest added vertex (u) is not 0 (is a real edge), and not visited yet
                visited.add(v) #visit it
                q.append(v) #add to queue for exploration
    
    assert len(visited) == n_v #assert that all nodes and no more were visited in MST.
                


