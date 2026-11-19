'''
Created on September 05, 2025

@NOTE: This file contains related implementation for BOTH "similarity computation" and "data preparation for graph embedding"
       1- LINOW (link_type: in-links or out-links)
       2- LINOW* (link_type: both-links)
       3- Computation on CPU: BOTH similarity computation and selecting TopK nodes are run in CPU with multi-processing
       4- Computation on GPU: BOTH similarity computation and selecting TopK nodes are run in GPU
'''

import numpy as np
import math
import tensorflow as tf
import os
import fetch_topK_SPARSE_BCH as prData
from scipy.special import softmax
import pickle

from tqdm import tqdm
from scipy.sparse import csr_matrix, csr_array
from sklearn.preprocessing import normalize
from concurrent.futures import ProcessPoolExecutor

def LINOW_bn_EMB(adj_mat=None, gr_type='', avg_deg=None, in_degrees=None, dataset_name='', iterations=5, damping_factor=0.2,
                 bch_size_cpu=128, bch_size_gpu=128, prl_num=8, GPU=False, scaling_factor=0, loss='',
                 write_topK_nodes=False, topK_save_path='', link_type="in-links", lambda_=0.9):
    '''
        implementation of LINOW and LINOW*
        @param adj_mat: graph adjacency matrix
        @param gr_type: graph type
        @param avg_deg: average degree of nodes
        @param in_degrees: in-degree of nodes
        @param dataset_name: graph name
        @param iterations: number of iteration
        @param damping_factor: C
        @param bch_size_cpu: batch size for CPU computation
        @param bch_size_gpu: batch size for GPU computation
        @param prl_num: number of processes for parallel computation
        @param GPU: flag for using GPU
        @param scaling_factor: The scaling factor t to compute topK
        @param loss: the target loss function
        @param write_topK_nodes: flag to save the topK nodes in a file
        @param topK_save_path: address to save the topK in a file
        @param link_type: the type of links considered in similarity computation
        @param lambda_: the in-links traversal probability for LINOW*
    '''
    print("Batch similarity computation --> dataset:'{}', iterations:{}, C:{}, batch:{}, {}, {}, for {}".format(
        dataset_name, iterations, damping_factor, (bch_size_gpu if GPU else bch_size_cpu),
        ('process:#' + str(prl_num) if not GPU else 'single process'), ('GPU' if GPU else 'CPU'), link_type) + '\n')

    num_nodes = adj_mat.shape[0]
    degrees = in_degrees
    out_degrees = adj_mat.sum(axis=1)

    if gr_type == 'directed':
        topK = scaling_factor * avg_deg
        if link_type == 'in-links': ## Computing LINOW based on in-links
            pass
        
        elif link_type == 'out-links': ## Computing LINOW based on out-links
            adj_mat = adj_mat.T
            degrees = out_degrees
            
        elif link_type == 'both-links': ## Computing LINOW*
            print("In-link traversal probability, λ:", lambda_)
            return LINOW_STAR_bn_EMB(adj_mat, dataset_name, iterations, damping_factor, bch_size_cpu, bch_size_gpu,
                                           prl_num, GPU, scaling_factor, loss, write_topK_nodes, topK_save_path, topK, lambda_)
        else:
            print('The link type is incorrect ...')
            exit()
    else:
        print('Graph is undirected; "link type" does not affect the embedding result ...')
        topK = scaling_factor * 2 * avg_deg
    print('# of topK nodes:', format(topK, ","), '\n')

    ####################################################
        # Prepare weight matrix
    ####################################################
    weights = csr_matrix(1 / np.log(degrees + math.e))
    weight_matrix = csr_matrix(adj_mat.multiply(weights)).astype(dtype='float32', casting='unsafe')
    del weights
    weight_matrix = normalize(weight_matrix, norm='l1', axis=0)
    weight_matrix = (damping_factor / 2) * weight_matrix
    top_indices = []

    ####################################################
        # GPU Computation
    ####################################################
    if GPU:
        batches = [range(i, min(i + bch_size_gpu, num_nodes)) for i in range(0, num_nodes, bch_size_gpu)]
        rows, cols = csr_array.nonzero(weight_matrix)
        indices = list(zip(rows, cols))
        weight_matrix_tensor = tf.sparse.SparseTensor(indices=indices, values=weight_matrix.data, dense_shape=[num_nodes, num_nodes])
        for idx in tqdm(range(0, len(batches)), smoothing=0.1):
            result_ = LINOW_bn_tensor_based_EMB(tf.constant(num_nodes, dtype=tf.int32), tf.constant(iterations, dtype=tf.int32),
                                                tf.constant(damping_factor, dtype=tf.float32), weight_matrix_tensor,
                                                tf.constant(batches[idx], dtype=tf.int32), tf.constant(loss, dtype=tf.string),
                                                tf.constant(topK, dtype=tf.int32))
            with tf.device('/CPU:0'):
                top_indices.extend(result_.numpy().tolist())

    ####################################################
        # CPU Computation
    ####################################################
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        batches = [range(i, min(i + bch_size_cpu, num_nodes)) for i in range(0, num_nodes, bch_size_cpu)]
        for idx in tqdm(range(0, len(batches), prl_num)):
            with ProcessPoolExecutor(max_workers=prl_num) as executor:
                futures = [executor.submit(LINOW_bn_pythonic_EMB, num_nodes, iterations, damping_factor, weight_matrix, batch, loss, topK) for batch in batches[idx:idx + prl_num]]
                for i, future in enumerate(futures):
                    top_indices.extend(future.result().tolist())

    ####################################################
        # Save the results if requested
    ####################################################
    name_ = topK_save_path + dataset_name + '_' + link_type + '_' + loss + '_Scl' + str(scaling_factor) + '_Iter' + str(iterations) + '_Dmp' + str(damping_factor)
    if write_topK_nodes:
        if not os.path.exists(topK_save_path):
            os.makedirs(topK_save_path)
        with open(name_, 'wb') as file_:
            pickle.dump(top_indices, file_)
        file_.close()
        print('The topK nodes are saved into a binary file ...')

    ####################################################
        # Return the results
    ####################################################
    if loss in ['listMLE', 'AttentionRank', 'listMLE_topK', 'SoftSort']:
        return tf.constant(top_indices, dtype=tf.int32)
    if loss == 'ListNET':
        return tf.constant(softmax(top_indices, axis=1), dtype=tf.float32)


def LINOW_STAR_bn_EMB(adj_mat=None, dataset_name='', iterations=0, damping_factor=0.8, bch_size_cpu=16, bch_size_gpu=16,
                            prl_num=1, GPU=False, scaling_factor=0, loss='', write_topK_nodes=False, topK_save_path='', topK=0, lambda_=0.9):
    '''
        Computes similarity and selecting topK nodes batch-wise with LINOW* 
        
        @param adj_mat: the graph dataset with size |V|*|V|
        @param iterations: number of iteration
        @param damping_factor: C
        @param bch_size_cpu: batch size for CPU computation
        @param bch_size_gpu: batch size for GPU computation
        @param prl_num: number of processes for parallel computation
        @param GPU: flag for using GPU
        @param scaling_factor: The scaling factor t to compute topK
        @param loss: the target loss function

    '''
    print('# of topK nodes:', topK, '\n')
    num_nodes = adj_mat.shape[0]

    in_degrees = adj_mat.sum(axis=0).T
    in_weights = csr_matrix(1 / np.log(in_degrees + math.e))
    in_weight_matrix = csr_matrix(adj_mat.multiply(in_weights)).astype(dtype='float32', casting='unsafe')
    del in_degrees
    del in_weights

    out_degrees = adj_mat.sum(axis=1).T
    out_weights = csr_matrix(1 / np.log(out_degrees + math.e))
    out_weight_matrix = csr_matrix(adj_mat.multiply(out_weights)).astype(dtype='float32', casting='unsafe')
    del out_degrees
    del out_weights
    del adj_mat

    in_weight_matrix = normalize(in_weight_matrix, norm='l1', axis=0)
    out_weight_matrix = normalize(out_weight_matrix, norm='l1', axis=1)

    random_srf_mat = lambda_ * in_weight_matrix + (1 - lambda_) * out_weight_matrix.T
    del in_weight_matrix
    del out_weight_matrix
    random_srf_mat = normalize(random_srf_mat, norm='l2', axis=0)
    random_srf_mat = (damping_factor / 2) * random_srf_mat
    top_indices = []

    ####################################################
        # GPU Computation
    ####################################################
    if GPU:
        batches = [range(i, min(i + bch_size_gpu, num_nodes)) for i in range(0, num_nodes, bch_size_gpu)]
        rows, cols = csr_array.nonzero(random_srf_mat)
        indices = list(zip(rows, cols))
        weight_matrix_tensor = tf.sparse.SparseTensor(indices=indices, values=random_srf_mat.data, dense_shape=[num_nodes, num_nodes])
        for idx in tqdm(range(0, len(batches)), smoothing=0.1):
            result_ = LINOW_bn_tensor_based_EMB(tf.constant(num_nodes, dtype=tf.int32), tf.constant(iterations, dtype=tf.int32),
                                                tf.constant(damping_factor, dtype=tf.float32), weight_matrix_tensor,
                                                tf.constant(batches[idx], dtype=tf.int32), tf.constant(loss, dtype=tf.string),
                                                tf.constant(topK, dtype=tf.int32))
            with tf.device('/CPU:0'):
                top_indices.extend(result_.numpy().tolist())

    ####################################################
        # CPU Computation
    ####################################################
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        batches = [range(i, min(i + bch_size_cpu, num_nodes)) for i in range(0, num_nodes, bch_size_cpu)]
        for idx in tqdm(range(0, len(batches), prl_num)):
            with ProcessPoolExecutor(max_workers=prl_num) as executor:
                futures = [executor.submit(LINOW_bn_pythonic_EMB, num_nodes, iterations, damping_factor, random_srf_mat, batch, loss, topK) for batch in batches[idx:idx + prl_num]]
                for i, future in enumerate(futures):
                    top_indices.extend(future.result().tolist())

    ####################################################
        # Save the results if requested
    ####################################################
    name_ = topK_save_path + dataset_name + '_Both_' + loss + '_Scl' + str(scaling_factor) + '_Iter' + str(iterations) + '_Dmp' + str(damping_factor)
    if write_topK_nodes:
        if not os.path.exists(topK_save_path):
            os.makedirs(topK_save_path)
        with open(name_, 'wb') as file_:
            pickle.dump(top_indices, file_)
        file_.close()
        print('The topK nodes are saved into a binary file ...')

    ####################################################
        # Return the results
    ####################################################
    if loss in ['listMLE', 'AttentionRank', 'listMLE_topK', 'SoftSort']:
        return tf.constant(top_indices, dtype=tf.int32)
    if loss == 'ListNET':
        return tf.constant(softmax(top_indices, axis=1), dtype=tf.float32)


def LINOW_bn_pythonic_EMB(n, K, C, W, q, loss, topK):
    """
        Pythonic implementation for batch similarity computation and data preparation for loss function

        @param n: number of nodes
        @param K: number of iterations
        @param C: damping factor
        @param W: column normalized weight matrix
        @param q: range of query nodes
        @param loss: the loss function
        @param topK: number of top similar nodes
        @return: numpy.ndarray in size |batch|*? based on the loss function
    """
    e_u = np.zeros((n, len(q)), dtype=np.float32)
    for i, query_node in enumerate(q):
        e_u[query_node, i] = 1

    zero = np.zeros((n, len(q)), dtype=np.float32)
    gamma = np.full((K + 2, K + 1), None, dtype=np.ndarray)
    for i in range(1, K + 2):
        gamma[i - 1, i - 1] = zero
        gamma[i, 0] = W.dot(gamma[i - 1, 0]) + e_u
        for j in range(1, i):
            gamma[i, j] = W.dot(gamma[i - 1, j]) + gamma[i - 1, j - 1]
            gamma[i - 1, j - 1] = None

    Gamma = e_u
    for i in range(1, K + 1):
        Gamma = gamma[K + 1, K - i] + W.T.dot(Gamma)

    if loss in ['listMLE', 'AttentionRank']:
        return prData.get_listMLE_BCH_py_v01(((1 - C) * Gamma).T)
    elif loss in ['ListNET']:
        return prData.get_listMLE_BCH_py(((1 - C) * Gamma).T)
    elif loss == 'listMLE_topK':
        return prData.get_listMLE_topK_BCH_py(((1 - C) * Gamma).T, topK)
    elif loss == 'SoftSort':
        return prData.get_SoftSort_BCH_py(((1 - C) * Gamma).T)


def LINOW_bn_tensor_based_EMB(n, K, C, W, q, loss, topK):
    """
        Tensor-based implementation for batch similarity computation and data preparation for loss function

        @param n: number of nodes
        @param K: number of iterations
        @param C: damping factor
        @param W: column normalized weight matrix
        @param q: range of query nodes
        @param loss: the loss function
        @param topK: number of top similar nodes
        @return: Tensor in size |batch|*? based on the loss function
    """
    e_u = tf.Variable(tf.zeros([n, len(q)]), dtype=np.float32)
    row_idx = q
    col_idx = tf.range(tf.shape(q)[0])
    indices = tf.stack([row_idx, col_idx], axis=1)
    updates = tf.ones(tf.shape(row_idx), dtype=tf.float32)
    e_u.assign(tf.tensor_scatter_nd_update(e_u, indices, updates))
    zero = tf.zeros((n, q.shape[0]), dtype=tf.float32)

    gamma = tf.TensorArray(dtype=tf.float32, size=(K + 2) * (K + 1), infer_shape=False, dynamic_size=True, clear_after_read=False)
    for i in range(1, K + 2):
        gamma = gamma.write((i - 1) * (K + 1) + (i - 1), zero)
        gamma = gamma.write(i * (K + 1), tf.sparse.sparse_dense_matmul(W, gamma.read((i - 1) * (K + 1))) + e_u)
        for j in range(1, i):
            gamma = gamma.write(i * (K + 1) + j, tf.sparse.sparse_dense_matmul(W, gamma.read((i - 1) * (K + 1) + j)) + gamma.read((i - 1) * (K + 1) + (j - 1)))
            gamma = gamma.write((i - 1) * (K + 1) + (j - 1), 0.)

    Gamma = e_u
    for i in tf.range(1, K + 1):
        Gamma = gamma.read((K + 1) * (K + 1) + (K - i)) + tf.sparse.sparse_dense_matmul(tf.sparse.transpose(W), Gamma)

    if loss in ['listMLE', 'AttentionRank']:
        return prData.get_listMLE_BCH_tb_v01(tf.transpose(tf.math.scalar_mul(1 - C, Gamma)))
    elif loss in ['ListNET']:
        return prData.get_listMLE_BCH_tb(tf.transpose(tf.math.scalar_mul(1 - C, Gamma)))
    elif loss == 'listMLE_topK':
        return prData.get_listMLE_topK_BCH_tb(tf.transpose(tf.math.scalar_mul(1 - C, Gamma)), topK)
    elif loss == 'SoftSort':
        return prData.get_SoftSort_BCH_tb(tf.transpose(tf.math.scalar_mul(1 - C, Gamma)))


if __name__ == "__main__":
    pass

    