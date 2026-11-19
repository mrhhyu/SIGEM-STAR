'''
AdaSim* similarity computation 
[GELTOR: A Graph Embedding Method based on Listwise Learning to Rank, WWW'23]
'''
import numpy as np
import math
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import fetch_topK_SPARSE as prData


def compute_AdaSim_star(adj_mat=None, gr_type='', avg_deg=None, in_degrees=None,
                        iterations=0, damping_factor=0.8, scaling_factor=0, loss=''):
    '''
        @param adj_mat: graph adjacency matrix
        @param gr_type: graph type (directed/undirected)
        @param avg_deg: average degree of nodes
        @param in_degrees: in-degree of nodes
        @param iterations: number of iterations
        @param damping_factor: C
        @param scaling_factor: The scaling factor t to compute topK
        @param loss: the target loss function
    '''
    num_nodes = adj_mat.shape[0]
    degrees = in_degrees

    if gr_type == 'directed':
        topK = scaling_factor * avg_deg
    else:
        topK = scaling_factor * 2 * avg_deg
    print('# of topK nodes:', format(topK, ","), '\n')

    weights = csr_matrix(1 / np.log(degrees + math.e))
    weight_matrix = csr_matrix(adj_mat.multiply(weights))
    del weights

    ####################################################
        # Adamic for k=1
    ####################################################
    print('Iteration 1 ...')
    adamic_scores = weight_matrix + weight_matrix.T + damping_factor * weight_matrix.T * adj_mat
    adamic_scores.setdiag(0)
    adamic_scores = adamic_scores / np.max(adamic_scores)

    result_matrix = 0.5 * adamic_scores
    result_matrix.setdiag(1)

    if iterations == 1:
        if loss == 'listMLE_topK':
            return prData.get_listMLE_topK(result_matrix, topK)

    weight_matrix = normalize(weight_matrix, norm='l1', axis=0)
    for itr in range(2, iterations + 1):
        print("Iteration " + str(itr) + ' ...')
        result_matrix.setdiag(0)
        temp = result_matrix * weight_matrix
        result_matrix = 0.5 * adamic_scores + damping_factor / 2.0 * (temp + temp.T)

    result_matrix.setdiag(1)
    if loss == 'listMLE_topK':
        return prData.get_listMLE_topK(result_matrix, topK)


if __name__ == "__main__":
    pass
