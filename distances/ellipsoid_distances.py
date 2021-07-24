import numpy as np
from itertools import combinations
from helper_functions import euclidean_distance_matrix

def SVD_ellipsoid_distances(data):
    n, d = data.shape
    Ms = [data[idx] for idx in indices]
    # in U, Sigma, V.T = SVD(M) get V.T
    Vs = [np.linalg.svd(M, full_matrices=True)[-1] for M in Ms]

    sigma = .05
    s = np.ones(d)
    s[-1] = sigma
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