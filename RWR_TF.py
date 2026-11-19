'''
The batch-wise CPU and GPU implementation of Random Walk with Restart (RWR) 
[Fast random walk with restart and its applications, ICDM'06]  
'''
import tensorflow as tf
import numpy as np
from scipy.sparse import csr_array, csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm
import fetch_topK_SPARSE_BCH as prData


def compute_RWR(adj_mat=None, gr_type='', avg_deg=None, iterations=0, alpha=0.15, scaling_factor=0,
                GPU=True, bch_size_cpu=1, bch_size_gpu=1, loss=''):
    '''
        @param adj_mat: graph adjacency matrix
        @param gr_type: graph type (directed/undirected)
        @param avg_deg: average degree of nodes
        @param iterations: number of iterations
        @param alpha: restart probability (fixed to 0.15)
        @param scaling_factor: The scaling factor t to compute topK
        @param GPU: flag for using GPU
        @param bch_size_cpu: batch size for CPU computation
        @param bch_size_gpu: batch size for GPU computation
        @param loss: the target loss function
    '''
    num_nodes = adj_mat.shape[0]

    if gr_type == 'directed':
        topK = scaling_factor * avg_deg
    else:
        topK = scaling_factor * 2 * avg_deg
    print('# of topK nodes:', format(topK, ","), '\n')

    adj_normalized = normalize(adj_mat, norm='l1', axis=1).astype(dtype='float32', casting='unsafe')
    top_indices = []

    ####################################################
        # GPU Computation
    ####################################################
    if GPU:
        batches = [range(i, min(i + bch_size_gpu, num_nodes)) for i in range(0, num_nodes, bch_size_gpu)]
        rows, cols = csr_array.nonzero(adj_normalized)
        indices = list(zip(rows, cols))
        adj_normalized_tensor = tf.sparse.SparseTensor(indices=indices, values=adj_normalized.data, dense_shape=[num_nodes, num_nodes])
        for idx in tqdm(range(0, len(batches)), smoothing=0.1):
            result_ = RWR_single_batch_Tensor(adj_normalized_tensor, tf.constant(batches[idx], dtype=tf.int32),
                                              tf.constant(iterations, dtype=tf.int32),
                                              tf.constant(alpha, dtype=tf.float32),
                                              tf.constant(topK, dtype=tf.int32),
                                              tf.constant(loss, dtype=tf.string))
            with tf.device('/CPU:0'):
                top_indices.extend(result_.numpy().tolist())

    ####################################################
        # CPU Computation
    ####################################################
    else:
        batches = [range(i, min(i + bch_size_cpu, num_nodes)) for i in range(0, num_nodes, bch_size_cpu)]
        for idx in tqdm(range(0, len(batches)), smoothing=0.1):
            result_ = RWR_single_batch(adj_normalized=adj_normalized, batch=batches[idx],
                                       alpha=alpha, max_iter=iterations, topK=topK, loss=loss)
            top_indices.extend(result_.tolist())

    if loss in ['listMLE_topK']:
        return tf.constant(top_indices, dtype=tf.int32)


def RWR_single_batch(adj_normalized='', batch=None, max_iter=0, alpha=0.15, topK=0, loss=''):
    '''
        Compute RWR scores for a single batch of nodes with CPU

        @param adj_normalized: normalized adjacency matrix
        @param batch: batch of query nodes
        @param alpha: restart probability
        @param max_iter: max iterations
        @param topK: number of top similar nodes
        @param loss: the loss function
    '''
    q = np.zeros((len(batch), adj_normalized.shape[0]), dtype=np.float32)
    q[:, batch[0]:batch[-1]+1] = np.eye(len(batch), dtype=np.float32)
    R = q
    for _ in range(max_iter):
        R = alpha * q + (1 - alpha) * R @ adj_normalized

    if loss in ['listMLE_topK']:
        return prData.get_listMLE_topK_BCH_py(R, topK)


def RWR_single_batch_Tensor(adj_normalized='', batch=None, max_iter=0, alpha=0.15, topK=0, loss=''):
    '''
        Compute RWR scores for a single batch of nodes with GPU

        @param adj_normalized: normalized adjacency matrix (SparseTensor)
        @param batch: batch of query nodes
        @param alpha: restart probability
        @param max_iter: max iterations
        @param topK: number of top similar nodes
        @param loss: the loss function
    '''
    q = tf.Variable(tf.zeros([len(batch), adj_normalized.shape[0]]), dtype=tf.float32)
    eye_sub = tf.eye(len(batch), dtype=tf.float32)
    q[:, batch[0]:batch[-1]+1].assign(eye_sub)
    R = q
    for _ in range(max_iter):
        R = alpha * q + (1 - alpha) * tf.sparse.sparse_dense_matmul(R, adj_normalized)

    if loss in ['listMLE_topK']:
        return prData.get_listMLE_topK_BCH_tb(R, topK)


if __name__ == "__main__":
    pass
