'''
    Created on April 17, 2026
    The Python implementation of SIGEM variants: SIGEM-ATR, SIGEM-MLE, SIGEM-NET, and SIGEM-SST in the paper  
    Implementation of: 
        AttentionRank [Learning a deep listwise context model for ranking refinement, SIGIR'18]; set loss_type='AttentionRank'
        ListMLE [Listwise approach to learning to rank: Theory and algorithm, ICML'08]; set loss_type='listMLE'
        ListNET [Learning to rank: from pairwise approach to listwise approach, ICML'07]; set loss_type='listNET'
        SoftSort [Softsort: A continuous relaxation for the argsort operator, ICML'20]; set loss_type='SoftSort'
'''
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers
import tensorflow.keras.backend as tkb
from argparse import ArgumentParser
from LINOW_EMB import LINOW_bn_EMB
from RWR_TF import compute_RWR
from AdaSim_star import compute_AdaSim_star
import time
import pickle
import numpy as np
import math
from scipy.sparse import csr_matrix

#---------- Wtih LARGE GRAPH comment the two following lines 
# if tf.config.list_physical_devices('GPU'):
#     tf.keras.mixed_precision.set_global_policy('mixed_float16')

class CustomCallback_verbose_check():
    '''
        1- Representing custom verbose information
        2- Applying dynamic learning rate
        3- Applying the early stopping
    '''
    def __init__(self, model, optimizer, early_stop, patience, min_delta):
        super().__init__()
        self.optimizer = optimizer
        self.model = model
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.best_weights = None
        self.stop_training = False
        self.best_epoch = None

    def on_epoch_begin(self, epoch):
        if not hasattr(self.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        cr_lr = float(tkb.get_value(self.optimizer.learning_rate))
        if 0 < epoch < 100 and (epoch % 10) == 0:
            new_lr = cr_lr - (0.1 * cr_lr)
            if new_lr > 1e-6:
                self.optimizer.learning_rate.assign(new_lr)

    def on_epoch_end(self, epoch_loss, epoch):
        print("Epoch {}, lr {}, Loss {}".format(epoch+1, round(float(tkb.get_value(self.optimizer.learning_rate)), 5), round(float(epoch_loss.numpy()), 4)))
        if epoch_loss < self.best_loss - self.min_delta:
            self.best_loss = epoch_loss.numpy()
            self.wait = 0
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
        elif self.early_stop:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                self.model.set_weights(self.best_weights)
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print("Model's weights are restored to the best ones at epoch {} with loss {}.".format(self.best_epoch, round(float(self.best_loss), 4)))

    def on_train_end(self):
        self.model.set_weights(self.best_weights)
        print("Training is finished; model's weights are restored to the best ones in epoch {}; .... Final loss: {}".format(self.best_epoch, round(float(self.best_loss), 4)))


class CustomDenseLayer(layers.Layer):
    '''
        A custom layer in shape d*|V|;
        Weight matrix d*|V| contains the latent vectors (column i is v_i)
    '''
    def __init__(self, input_shape, output_shape, reg_rate, **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.in_shape = input_shape
        self.out_shape = output_shape
        self.reg_rate = reg_rate

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.in_shape, self.out_shape),
            regularizer=regularizers.L2(self.reg_rate),
            name='kernel')

    def call(self, inputs):
        x = tf.gather(self.kernel, inputs, axis=1)
        x = tf.matmul(tf.transpose(x), self.kernel)
        return tf.nn.selu(x)


class SIGEM_model(Model):
    '''
        Single layer model
    '''
    def __init__(self, dim, num_nodes, reg_rate):
        super(SIGEM_model, self).__init__()
        self.layer_0 = CustomDenseLayer(dim, num_nodes, reg_rate, name='layer_0')

    @tf.function
    def call(self, inputs):
        return self.layer_0(inputs)


####################################################
# Loss Functions
####################################################
def compute_loss_listMLE(y_true, y_pred):
    '''Full listMLE loss function'''
    sorted_y_pred = tf.gather(y_pred, y_true, axis=1, batch_dims=1)
    raw_max = tf.reduce_max(input_tensor=sorted_y_pred, axis=1, keepdims=True)
    sorted_y_pred = sorted_y_pred - raw_max
    cumsum_sums = tf.cumsum(tf.exp(sorted_y_pred), axis=1, reverse=True)
    final_sums = tf.math.log(cumsum_sums + tf.keras.backend.epsilon()) - sorted_y_pred
    return tf.reduce_mean(tf.reduce_sum(input_tensor=final_sums, axis=1, keepdims=True))

def compute_loss_AttentionRank(y_true, y_pred):
    '''AttentionRank loss function'''
    y_true_labels = tf.range(0., -tf.cast(tf.shape(y_true)[1], tf.float32), -1., dtype=tf.float32)
    y_true_labels = tf.exp(y_true_labels)
    sum_all_y_true_labels = tf.reduce_sum(y_true_labels)
    attention_y_true_labels = tf.math.divide(y_true_labels, sum_all_y_true_labels)

    y_true_scores = tf.gather(y_pred, y_true, axis=1, batch_dims=1)
    max_ = tf.reduce_max(y_true_scores, axis=1, keepdims=True)
    y_true_scores = y_true_scores - max_
    y_true_scores = tf.exp(y_true_scores)
    sum_all_y_true_scores = tf.reduce_sum(y_true_scores, axis=1, keepdims=True)
    attention_y_true_scores = tf.math.divide(y_true_scores, sum_all_y_true_scores)

    loss_values = -(tf.math.multiply(attention_y_true_labels, tf.math.log(attention_y_true_scores + tf.keras.backend.epsilon())) +
                    tf.math.multiply(1.0 - attention_y_true_labels, tf.math.log(1.0 - attention_y_true_scores + tf.keras.backend.epsilon())))
    return tf.reduce_mean(tf.reduce_sum(input_tensor=loss_values, axis=1, keepdims=True))

def compute_loss_listNet(y_true, y_pred):
    '''ListNET loss function'''
    max_ = tf.reduce_max(y_pred, axis=1, keepdims=True)
    y_pred_exp = tf.exp(y_pred - max_)
    y_pred_exp_sum = tf.reduce_sum(y_pred_exp, axis=1, keepdims=True)
    y_pred_softx = tf.divide(y_pred_exp, y_pred_exp_sum)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(tf.math.multiply(tf.cast(y_true, dtype=tf.float32), tf.math.log(y_pred_softx + tf.keras.backend.epsilon())), axis=1, keepdims=True))
    return cross_entropy

def compute_loss_SoftSort(y_true, y_pred):
    '''SoftSort loss function'''
    y_pred_fetched = tf.gather(y_pred, y_true, axis=1, batch_dims=1)
    y_pred_fetched = tf.expand_dims(y_pred_fetched, 2)
    y_pred_sorted = tf.sort(y_pred_fetched, direction='DESCENDING', axis=1)
    pairwise_distances = -tf.abs(tf.transpose(y_pred_fetched, perm=[0, 2, 1]) - y_pred_sorted)
    P_hat = tf.nn.softmax(pairwise_distances / 0.03)
    return tf.reduce_mean(-tf.math.log(tf.linalg.diag_part(P_hat) + tf.keras.backend.epsilon()))


class UnifiedLoss(tf.keras.losses.Loss):
    def __init__(self, loss_type='listMLE', name="unified_loss"):
        super().__init__(name=name)
        self.loss_type = loss_type

    def compute_loss(self, y_true, y_pred):
        if self.loss_type == 'listMLE':
            return compute_loss_listMLE(y_true, y_pred)
        elif self.loss_type == 'AttentionRank':
            return compute_loss_AttentionRank(y_true, y_pred)
        elif self.loss_type == 'ListNET':
            return compute_loss_listNet(y_true, y_pred)
        elif self.loss_type == 'SoftSort':
            return compute_loss_SoftSort(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


@tf.function
def train_one_epoch(model, loss_fn, optimizer, dataset):
    '''execute training step for one epoch'''
    epoch_loss = tf.constant(0.0)
    num_batches = tf.constant(0)
    for batch_x, batch_y in dataset:
        with tf.GradientTape() as tape:
            y_pred = model(batch_x)
            loss = loss_fn.compute_loss(batch_y, tf.cast(y_pred, tf.float32))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_loss += tf.cast(loss, tf.float32)
        num_batches += 1
    epoch_loss /= tf.cast(num_batches, tf.float32)
    return epoch_loss


def embedding(args):
    start = time.perf_counter()

    # Validation
    if not os.path.exists(args.graph):
        print('ERROR: graph is invalid ...!\n')
        exit()
    if not args.dataset_name:
        print('ERROR: please input the dataset name ...!\n')
        exit()
    if args.lr is None:
        print('ERROR: please set the learning rate; 0.0030 for small graphs and 0.0012 for very large graphs ...!\n')
        exit()
    if args.scaling_factor is None:
        print('ERROR: please set the scaling factor; 10 for link prediction and node classification, and 2 for graph reconstruction ...!\n')
        exit()
    if args.loss_type not in ['listMLE', 'AttentionRank', 'ListNET', 'SoftSort']:
        print('ERROR: please choose the loss funcitons as listMLE, AttentionRank, ListNET, or SoftSort...!\n')                                                                                                                    
        exit()   
        
    ####################################################
    # construct info string for output
    ####################################################
    if args.loss_type == 'listMLE':
        method_name = f'_SIGEM_MLE_'
    if args.loss_type == 'AttentionRank':
        method_name = f'_SIGEM_ATR_'        
    if args.loss_type == 'ListNET':
        method_name = f'_SIGEM_NET_'        
    if args.loss_type == 'SoftSort':
        method_name = f'_SIGEM_SST_'        
    if args.gpu:
        com_type = f'_GPU_bch{args.bch_gpu}_'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        com_type = f'_CPU_bch{args.bch_cpu}_process{args.prl_num}_'

    print()
    print(args, '\n')
    info = args.result_dir + args.dataset_name + method_name + 'IT' + str(args.itr) + '_C' + str(int(args.damping_factor * 10)) + com_type + 'lr' + str(args.lr).split('.')[1]\
     + '_dim' + str(args.dim) + '_Scl' + str(args.scaling_factor)

    ####################################################
    # Similarity Computation
    ####################################################
    if not args.read_topK_nodes:
        print(f"{'='*50} Similarity Computation {'='*50}\n")

        # Read graph in SIGEM
        edges = np.loadtxt(args.graph, dtype=int)
        num_nodes = np.max(edges) + 1
        num_edges = len(edges)
        avg_degree = math.ceil(num_edges / num_nodes)
        adj = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes), dtype='int32')
        print('Adjacency matrix is constructed ...')
        degrees = adj.sum(axis=0).T
        out_degrees = adj.sum(axis=1)
        if (degrees != out_degrees).any():
            print("Input graph is Directed ...")
            graph_type = 'directed'
            reg = 0.001
        else:
            print("Input graph is Undirected ...")
            reg = 0.00001
            graph_type = 'undirected'
        print("# of nodes:", num_nodes)
        print("# of edges:", num_edges)
        print("Average degree:", round(num_edges / num_nodes, 2))
        print("Scaling factor:", args.scaling_factor)

        top_indices = LINOW_bn_EMB(adj_mat=adj, gr_type=graph_type, avg_deg=avg_degree, in_degrees=degrees,
                                   dataset_name=args.dataset_name, iterations=args.itr, damping_factor=args.damping_factor,
                                   bch_size_cpu=args.bch_cpu, bch_size_gpu=args.bch_gpu,
                                   prl_num=args.prl_num, GPU=args.gpu, scaling_factor=args.scaling_factor,
                                   loss=args.loss_type,
                                   write_topK_nodes=args.write_topK_nodes, topK_save_path=args.topK_save_path,
                                   link_type='in-links')
    else:
        print('Loading topK nodes from a binary file ...\n')
        with open(args.topK_file, 'rb') as file_:
            top_indices = pickle.load(file_)
            top_indices = tf.constant(top_indices, dtype=tf.int32)
        reg = 0.001

    ####################################################
    # Create input tensor
    ####################################################
    print(f"{'='*50} Model Training {'='*50}\n")
    tf_input = tf.range(len(top_indices))

    ####################################################
    # Create batches for training
    ####################################################
    print('Creating batches for training ... ')
    dataset = tf.data.Dataset.from_tensor_slices((tf_input, top_indices))
    if args.gpu:
        dataset = (dataset
                   .shuffle(buffer_size=2048)
                   .batch(args.bch_gpu)
                   .cache()
                   .prefetch(tf.data.AUTOTUNE)
                   .with_options(tf.data.Options()))
    else:
        dataset = (dataset
                   .shuffle(buffer_size=2048)
                   .batch(args.bch_cpu)
                   .cache()
                   .prefetch(tf.data.AUTOTUNE)
                   .with_options(tf.data.Options()))

    ####################################################
    # Initializing model
    ####################################################
    model = SIGEM_model(args.dim, len(top_indices), reg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=0.5)
    loss_fn = UnifiedLoss(loss_type=args.loss_type)
    model.build((None,))
    cc_verbose_check = CustomCallback_verbose_check(model, optimizer, args.early_stop, args.wait_thr, 0)

    ####################################################
    # Training
    ####################################################
    print('Training started ...\n')
    for epoch in range(args.epc):
        cc_verbose_check.on_epoch_begin(epoch)
        epoch_loss = train_one_epoch(model, loss_fn, optimizer, dataset)
        cc_verbose_check.on_epoch_end(epoch_loss, epoch)
        if cc_verbose_check.stop_training:
            break
    if not cc_verbose_check.stop_training:
        cc_verbose_check.on_train_end()

    ####################################################
    # Write results
    ####################################################
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    stop = time.perf_counter()
    time_file = open(info + '_time.txt', 'w')
    time_file.write("Training time: " + str(round((stop - start) / 60, 3)) + '\n')
    time_file.close()

    np.save(info + '_emb', model.get_layer('layer_0').weights[0][:].numpy().T)
    print('The embedding result is written in the file .... \n')


def parse_args(graph='', dataset_name='', result_dir='', dimension=128, iterations=5, damping_factor=0.2, scaling_factor=None,
               gpu_on=True, bch_cpu=32, bch_gpu=32, prl_num=8, epochs=100,
               learning_rate=None, early_stop=True, wait_thr=15,
               read_topK_nodes=False, topK_file='', write_topK_nodes=False, topK_save_path='',
               loss_type='listMLE'):

    parser = ArgumentParser(description="SIGEM - Unified Graph Embedding with multiple similarity methods and loss types.")
    parser.add_argument('--graph', nargs='?', default=graph, help='Input graph')
    parser.add_argument('--dataset_name', nargs='?', default=dataset_name, help='Dataset name')
    parser.add_argument('--result_dir', nargs='?', default=result_dir, help='Destination to save the embedding result')
    parser.add_argument('--loss_type', type=str, choices=['listMLE_topK', 'listMLE', 'AttentionRank', 'ListNET', 'SoftSort'],
                        default=loss_type, help='Loss function type')
    parser.add_argument('--dim', type=int, default=dimension, help='Dimensionality of embeddings')
    parser.add_argument('--itr', type=int, default=iterations, help='Number of iterations to compute similarity scores')
    parser.add_argument('--damping_factor', type=float, default=damping_factor, help='Damping factor for LINOW')
    parser.add_argument('--scaling_factor', type=int, default=scaling_factor, help='Scaling factor to compute topK; 10 for link prediction and node classification, 2 for graph reconstruction')
    parser.add_argument('--gpu', type=bool, default=gpu_on, help='Flag to run computation on GPU')
    parser.add_argument('--bch_cpu', type=int, default=bch_cpu, help='Batch size for CPU computation; for SoftSort set is to a very small value')
    parser.add_argument('--bch_gpu', type=int, default=bch_gpu, help='Batch size for GPU computation; for SoftSort set is to 1')
    parser.add_argument('--prl_num', type=int, default=prl_num, help='Number of parallel processes on CPU')
    parser.add_argument('--epc', type=int, default=epochs, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=learning_rate, help='Learning rate')
    parser.add_argument('--early_stop', type=bool, default=early_stop, help='Enable early stopping')
    parser.add_argument('--wait_thr', type=int, default=wait_thr, help='Patience for early stopping')
    parser.add_argument('--read_topK_nodes', type=bool, default=read_topK_nodes, help='Read topK from saved file')
    parser.add_argument('--topK_file', type=str, default=topK_file, help='Saved topK binary file path')
    parser.add_argument('--write_topK_nodes', type=bool, default=write_topK_nodes, help='Write topK to binary file')
    parser.add_argument('--topK_save_path', type=str, default=topK_save_path, help='Path to save topK binary file')

    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args(
        graph='/feature_learning/CoCit/dataset/train_test_50/CoCit_train_50.txt',
        dataset_name='CoCit_train_50',
        result_dir='../result_test/',
        dimension=128,
        scaling_factor=10,
        iterations=1,
        damping_factor=0.2,
        loss_type='ListNET', # listMLE, AttentionRank, ListNET, SoftSort
        epochs=2,
        bch_cpu=128,
        bch_gpu=128,
        gpu_on=True,
        prl_num=8,
        learning_rate=0.0030,
        early_stop=True,
        wait_thr=10,
    )
    embedding(args)
