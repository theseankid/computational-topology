import numpy as np
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from .helper_functions import euclidean_distance_matrix, jacobian_rotation_matrices, svd_rotation_matrices


def euclidean_distance_matrix(data, normalize = False, symmetric = True):
    """
    index i, j of D[i,j] is the Euclidean distance of row i and j of the data matrix

    data : a should be shape n by d array where n is the number of observations

    normalize : True puts distances to [0,1] range

    symmetric : False returns an upper triangular matrix, useful for sparse matrix representation and histograms
    """

    n = data.shape[0]
    D = np.zeros([n,n])
    
    for i, j in combinations(np.arange(n),2):
        D[i,j] = np.linalg.norm(data[i] - data[j])

    if symmetric:
        D = D + D.T

    if normalize:
        D = D/D.max()

    return D


def neighbors_distances(data):
    '''
    while not a proper distance metric, it is an input for calculating a filtration based on nearest neighbors  
    
    calculate distances by nearest neighbors of all the points
    for points i, j take the minimum as the distance
    '''

    n, d = data.shape
    assert n > 1, 'need to have more than one observation for distance calculation'

    nbrs = NearestNeighbors(n_neighbors=n).fit(data)
    # distance matrix for a filtration
    all_indices = nbrs.kneighbors(data)[-1]
    
    D = np.zeros(all_indices.shape)
    for i, row in enumerate(all_indices):    
        for idx, j in enumerate(row):
            D[i, j] = idx

    min_matrix = np.vectorize(min)

    return min_matrix(D, D.T)
