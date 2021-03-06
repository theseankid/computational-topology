import numpy as np
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from .helper_functions import euclidean_distance_matrix, jacobian_rotation_matrices, svd_rotation_matrices
import sympy
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


def jacobian_ellipsoid_distances(data, f, variables, sigma):
    '''
    f : function for finding tangent spaces of spheroids
    
    variables: a list of variables of sympy.Symbol type

    sigma : sets ratio for ellipsoid distances
    '''

    n, d = data.shape

    for var in variables:
        assert type(var) == type(sympy.Symbol('foo')), 'variable {} must be of sympy.Symbol type'.format(str(var))

    assert 0 < sigma <=1, 'sigma must be in (0, 1]'
    assert type(data) == np.ndarray, ' must be numpy array (matrix)'

    dists = euclidean_distance_matrix(data, normalize=True)

    if sigma == 1:
        return dists

    Vs = jacobian_rotation_matrices(data, f, variables)

    s = np.ones(d)
    s[0] = sigma

    # creates oblate spheroid mappings at each point
    Qs = [V.T*s@V for V in Vs]

    D = np.zeros([n,n])

    for i,j in combinations(np.arange(n),2):
        Qi = Qs[i]
        Qj = Qs[j]
        u, v = data[[i, j]]

        h = (u-v)/np.linalg.norm(u-v)
        a1 = h.T@Qs[i]@h
        a2 = h.T@Qs[j]@h

        D[i,j] = 2*dists[i,j]/(np.sqrt(a1)+np.sqrt(a2))

    return D + D.T


def svd_ellipsoid_distances(data, sigma=1.0, nnbrs=1, fixed=True):

    '''
    nnbrs : chooses subsets of data for calculating elipsoid distances

    fixed : False sets sigma = min singular value for each ellipsoid
    '''

    n, d = data.shape
    assert d > 1, 'need to have more than one dimension for SVD ellipsoids'
    assert nnbrs > 0, 'need to have more neighbors than dimension of data'
    assert 0 < sigma <=1, 'sigma must be in (0, 1]'

    dists = euclidean_distance_matrix(data, normalize=True)

    if sigma == 1:
        return dists 

    if fixed:
        Vs, Sigmas = svd_rotation_matrices(data, nnbrs, fixed)

        s = np.ones(d)
        s[-1] = sigma

        Qs = [V.T*s@V for V in Vs]

    else:
        Vs, Sigmas = svd_rotation_matrices(data, nnbrs, fixed)

        Qs = [V.T*s@V for s, V in zip(Sigmas, Vs)]

    D = np.zeros([n,n])

    for i,j in combinations(np.arange(n),2):
        Qi = Qs[i]
        Qj = Qs[j]
        u, v = data[[i, j]]

        h = (u-v)/np.linalg.norm(u-v)
        a1 = h.T@Qs[i]@h
        a2 = h.T@Qs[j]@h

        D[i,j] = 2*dists[i,j]/(np.sqrt(a1)+np.sqrt(a2))

    return D + D.T


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
