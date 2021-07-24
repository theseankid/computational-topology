import numpy as np

def euclidean_distance_matrix(data, normalize = False):
    n = data.shape[0]
    D = np.zeros([n,n])
    
    for i, j in combinations(np.arange(n),2):
        D[i,j] = np.linalg.norm(data[i] - data[j])
    
    if normalize:
        return (D + D.T)/D.max()
        
    else:
        return (D + D.T)
