
import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def generate_physical_coordinates(n):
    phy_coord = np.random.randn(n, 2)
    return phy_coord

def test_phy_coordinates(n):
    phy_coord = np.full((n, 2), 0)

    phy_coord[0][0] = 0
    phy_coord[0][1] = 0

    phy_coord[1][0] = 0
    phy_coord[1][1] = 1

    phy_coord[2][0] = 1
    phy_coord[2][1] = 1

    phy_coord[3][0] = 2
    phy_coord[3][1] = 1

    phy_coord[4][0] = 2
    phy_coord[4][1] = 2

    phy_coord[5][0] = 3
    phy_coord[5][1] = 2

    phy_coord[6][0] = 3
    phy_coord[6][1] = 3

    phy_coord[7][0] = 4
    phy_coord[7][1] = 3

    phy_coord[8][0] = 4
    phy_coord[8][1] = 4

    phy_coord[9][0] = 5
    phy_coord[9][1] = 4

    return phy_coord

def get_adj(phy_coord):
    adj = np.zeros((phy_coord.shape[0], phy_coord.shape[0]))
    
    for i in range(phy_coord.shape[0]):
        for j in range(phy_coord.shape[0]):
            if is_neighbour(i, j, phy_coord):
                adj[i][j] = 1
            else:
                adj[i][j] = 0

    return adj

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)
    rowsum = np.zeros(adj.shape[0])
    
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            rowsum[i] += adj[i][j]

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


# function ge_VC will be called by external modules to calculate VC
def distance(x, y, dist_matrix):
    return tf.math.sqrt((dist_matrix[x, 0] - dist_matrix[y, 0])**2 + (dist_matrix[x, 1] - dist_matrix[y, 1])**2) 
    
def is_neighbour(x, y, dist_matrix):
    return distance(x, y, dist_matrix) <= 1


# finding shortest paths based on Dynamic Programming
def get_shortest_path(source, dest, dist_matrix, VC_matrix):

    # return 1 + min distance from source to a neighbour of destination
    for _ in range(0, dist_matrix.shape[0]):
        for i in range(0, dist_matrix.shape[0]):
            if is_neighbour(source, i, dist_matrix):
                VC_matrix[source][dest] = min(VC_matrix[source][dest], 1 + get_shortest_path(i, dest, dist_matrix, VC_matrix))


    return VC_matrix[source][dest]

# dist_matrix is the matrix of geographical coordinates: Number_of_nodes X 2
# anchors is the list of anchor nodes: Number_of_anchors X 1

def calculate_shortest_hops(dist_matrix, anchors):
    VC_matrix = 1000 * np.ones((dist_matrix.shape[0], len(anchors)))

    for temp in range(len(anchors)):
        VC_matrix[anchors[temp]][temp] = 0

    for source in range(dist_matrix.shape[0]):
        for dest in range(len(anchors)):

            if VC_matrix[source][dest] > 0:
                # if these nodes are neighbours, then hop_dist =  1
                if(distance(source, anchors[dest], dist_matrix) <= 1):
                    VC_matrix[source][dest] = 1
                    # VC_matrix[dest][source] = VC_matrix[source][dest]

    # print('Prelim investigation done: {}'.format(VC_matrix))

    while not processed_all_nodes(VC_matrix, dist_matrix, anchors):
        for source in range(0, dist_matrix.shape[0]):
            for dest in range(0, len(anchors)):
                for i in range(dist_matrix.shape[0]):
                    if is_neighbour(source, i, dist_matrix):
                        VC_matrix[source][dest] = min(VC_matrix[source][dest], 1 + VC_matrix[i][dest])

    # print('Completed VC processing')
    return VC_matrix

def baked_rice(VC_matrix, dist_matrix, anchors):
    for source in range(0, dist_matrix.shape[0]):
            for dest in range(0, len(anchors)):
                if VC_matrix[source][dest] == 1000:
                    for i in range(dist_matrix.shape[0]):
                        if is_neighbour(source, i, dist_matrix):

                            if VC_matrix[source][dest] > 1 + VC_matrix[i][dest]:
                                VC_matrix[source][dest] = 1 + VC_matrix[i][dest]
                                return 0

    return 1

def processed_all_nodes(VC_matrix, dist_matrix, anchors):
    ''' Function to verify if processing is complete '''

    for source in range(0, dist_matrix.shape[0]):
            for dest in range(0, len(anchors)):
                for i in range(dist_matrix.shape[0]):
                    if is_neighbour(source, i, dist_matrix):

                        if VC_matrix[source][dest] > 1 + VC_matrix[i][dest]:
                            VC_matrix[source][dest] = 1 + VC_matrix[i][dest]
                            return 0

    return 1



def select_anchor_nodes(dist_matrix):
    anchor_list = np.random.choice(range(dist_matrix.shape[0]), 5, replace=False)

    return anchor_list

def get_VC(num_of_nodes):
    # dist_matrix = generate_physical_coordinates(num_of_nodes)
    dist_matrix = test_phy_coordinates(num_of_nodes)
    anchors = select_anchor_nodes(dist_matrix)

    print('DEBUG: Anchor list: {}'.format(anchors))
    return tf.convert_to_tensor(dist_matrix, dtype=tf.float64), tf.convert_to_tensor(calculate_shortest_hops(dist_matrix, anchors), dtype=tf.float64)
