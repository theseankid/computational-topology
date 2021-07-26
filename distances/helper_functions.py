import numpy as np
from itertools import combinations

def euclidean_distance_matrix(data, normalize = False, symmetric = True):
    """
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
        return D/D.max()

    else:
        return D
