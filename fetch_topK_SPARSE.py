'''
Fetches topK similar nodes to each node from a AdaSim* similarity matrix.
'''
import numpy as np

def get_listMLE_topK(result_matrix, topK):
    '''
        @return: top_indices: |V|*topK matrix contains indices of topK nodes to each node
    '''
    print("Fetching Top {} similar nodes started ... ".format(topK))
    top_indices = np.zeros((result_matrix.shape[0], topK), dtype='int32')
    for target_node in range(0, result_matrix.shape[0]):
        target_node_res_sorted = np.argsort(result_matrix[target_node, :].toarray()[0], axis=0)[::-1][:topK]
        top_indices[target_node] = target_node_res_sorted
    print("Top {} similar nodes are fetched ... ".format(topK))
    return top_indices
