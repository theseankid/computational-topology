import numpy as np
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from helper_functions import euclidean_distance_matrix
from sympy import lambdify
from sympy import Matrix


def ellipsoid_distances(data, f, variables, sigma):

    n, d = data.shape

    J = lambdify(variables, Matrix([f]).jacobian(variables))
    
    # Vs in one step
    Vs = [np.linalg.svd(Jf_lam(*row), full_matrices=True)[-1] for row in data]

    s = np.ones(d)
    s[0] = sigma
    # creates oblate spheroid mappings at each point
    Qs = [V.T*s@V for V in Vs]

    dists = euclidean_distance_matrix(data, normalize=True)
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


def SVD_ellipsoid_distances(data):

    n, d = data.shape
    assert d > 1, 'need to have more than one dimension for SVD ellipsoids'

    Ms = [data[idx] for idx in indices]
    # in U, Sigma, V.T = SVD(M) get V.T
    Vs = [np.linalg.svd(M, full_matrices=True)[-1] for M in Ms]

    sigma = .05
    s = np.ones(d)
    s[-1] = sigma

    # creates oblate spheroid mappings at each point
    Qs = [V.T*s@V for V in Vs]

    dists = euclidean_distance_matrix(data, normalize=True)
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


def neighbors_distance(data):

    n, d = data.shape
    assert n > 1, 'need to have more than one observation for distance calculation'

    nbrs = NearestNeighbors(n_neighbors=n).fit(data)
    # distance matrix for a ripser filtration
    all_indices = nbrs.kneighbors(data)[-1]
    
    D = np.zeros(all_indices.shape)
    for i, row in enumerate(all_indices):    
        for idx, j in enumerate(row):
            D[i, j] = idx
    return D/(n-1)
