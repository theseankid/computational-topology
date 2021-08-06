import numpy as np
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from sympy import lambdify
from sympy import Matrix

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
        return D/D.max()

    else:
        return D


def jacobian_rotation_matrices(data, f, variables):
    '''
    f : function for finding tangent spaces of spheroids
    
    variables: a list of variables of sympy.Symbol type
    '''

    Jf = lambdify(variables, Matrix([f]).jacobian(variables))

    # Vs in one step
    Vs = [np.linalg.svd(Jf(*row), full_matrices=True)[-1] for row in data]
    
    return Vs


def svd_rotation_matrices(data, nnbrs=1, fixed=True):

    n, d = data.shape

    nbrs = NearestNeighbors(n_neighbors=d+nnbrs).fit(data)
    distances, indices = nbrs.kneighbors(data)

    Ms = [data[idx] for idx in indices]

    # creates oblate spheroid mappings at each point
    if fixed:
        Vs = [np.linalg.svd(M, full_matrices=True)[-1] for M in Ms]

        return Vs, None

    else:
        Sigs, Vs = zip(*[np.linalg.svd(M, full_matrices=True)[-2:] for M in Ms])

        Sigmas = []
        for sigs in Sigs:
            s_ones = np.ones(d)
            s_ones[-1] = sigs[-1]/sigs.sum()
            Sigmas.append(s_ones)

        return Vs, Sigmas
