'''
Fetches topK similar nodes to each node from a similarity matrix.
Provides two implementations per method: pythonic (*_py) and tensor based (*_tb);
Former is used with CPU computation and latter with GPU computation.
'''
import numpy as np
import tensorflow as tf


def get_listMLE_topK_BCH_py(result_matrix, topK):
    '''
        @return: top_indices: |batch|*topK matrix contains indices of topK nodes to each target node
    '''
    if topK > result_matrix.shape[1]:
        top_indices = np.argsort(-result_matrix, axis=1)[:, :result_matrix.shape[1]]
    else:
        top_indices = np.argsort(-result_matrix, axis=1)[:, :topK]
    return top_indices


def get_listMLE_topK_BCH_tb(result_matrix, topK):
    '''
        @return: top_indices: |batch|*topK tensor contains descending order of nodes
    '''
    if topK > result_matrix.shape[1]:
        _, topk_indices = tf.nn.top_k(result_matrix, k=result_matrix.shape[1], sorted=True)
    else:
        _, topk_indices = tf.nn.top_k(result_matrix, k=topK, sorted=True)
    return topk_indices


def get_listMLE_BCH_py(result_matrix):
    '''
        @note: it is compatible with the implementation of ListMLE in tensorflow / usable for ListNET too
        @param result_matrix: numpy.ndarray in size |batch|*|V|
        @return: sorted_indices is a |batch|*|V| ndarray contains the rank of each node regarding to a target node;
                 node with highest similarity score has highest rank
                 e.g., result_matrix=[0.9,0.1,0.5] --> sorted_indices=[2,0,1]
    '''
    sorted_indices = np.zeros((result_matrix.shape[0], result_matrix.shape[1]), dtype='int32')
    for target_node in range(0, result_matrix.shape[0]):
        target_node_res_sorted = np.argsort(result_matrix[target_node, :], axis=0)
        rank = np.arange(result_matrix.shape[1], dtype='int32')
        low = min(rank.min(), target_node_res_sorted.min())
        high = max(rank.max(), target_node_res_sorted.max())
        lut = np.zeros(int(high - low + 1), dtype='float32')
        lut[target_node_res_sorted - low] = np.arange(len(target_node_res_sorted))
        sorted_indices[target_node] = lut[rank - low].copy()
    return sorted_indices


def get_listMLE_BCH_tb(result_matrix):
    '''
        @note: it is compatible with the implementation of ListMLE in tensorflow / usable for ListNET too
        @param result_matrix: tensor in size |batch|*|V|
        @return: sorted_indices is a |batch|*|V| tensor contains the rank of each node regarding to a target node;
                 node with highest similarity score has highest rank
                 e.g., result_matrix=[0.9,0.1,0.5] --> sorted_indices=[2,0,1]
    '''
    def rank_nodes(row):
        sorted_indices_row = tf.Variable(tf.zeros(result_matrix.shape[1], dtype=tf.int32))
        row_arg_sorted = tf.argsort(row, axis=0, direction='ASCENDING')
        rank = tf.range(result_matrix.shape[1], dtype=tf.int32)
        sorted_indices_row = tf.tensor_scatter_nd_update(sorted_indices_row, tf.expand_dims(row_arg_sorted, axis=1), rank)
        return sorted_indices_row
    sorted_indices = tf.map_fn(rank_nodes, result_matrix, fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32))
    return sorted_indices


def get_listMLE_BCH_py_v01(result_matrix):
    '''
        @param result_matrix: numpy.ndarray in size |batch|*|V|
        @return: top_indices: |batch|*|V| tensor contains descending order of nodes regarding to a target node;
                 e.g., result_matrix=[0.9,0.1,0.5] --> sorted_indices=[0,2,1]
    '''
    top_indices = np.argsort(-result_matrix, axis=1)
    return top_indices


def get_listMLE_BCH_tb_v01(result_matrix):
    '''
        @param result_matrix: tensor in size |batch|*|V|
        @return: top_indices: |batch|*|V| tensor contains descending order of nodes regarding to a target node;
                 e.g., result_matrix=[0.9,0.1,0.5] --> sorted_indices=[0,2,1]
    '''
    sorted_indices = tf.argsort(result_matrix, axis=-1, direction='DESCENDING')
    return sorted_indices


def get_SoftSort_BCH_py(result_matrix):
    '''
        prepares the required information for 'SoftSort' loss function
        @return: top_indices: |batch|*|V| matrix contains the argsort of all scores
    '''
    top_indices = np.zeros((result_matrix.shape[0], result_matrix.shape[1]), dtype='int32')
    for target_node in range(0, result_matrix.shape[0]):
        target_node_res_sorted = np.argsort(result_matrix[target_node, :], axis=0)[::-1]
        top_indices[target_node] = target_node_res_sorted
    return top_indices


def get_SoftSort_BCH_tb(result_matrix):
    '''
        prepares the required information for 'SoftSort' loss function
        @return: top_indices: |batch|*|V| tensor contains the argsort of all scores
    '''
    def argsort_nonzero(row):
        sorted_indices = tf.argsort(row, axis=0, direction='DESCENDING')
        return sorted_indices
    sorted_indices_per_row = tf.map_fn(argsort_nonzero, result_matrix, fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32))
    return sorted_indices_per_row
