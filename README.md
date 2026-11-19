# Simple is Better: A Simple yet Effective Graph Embedding Method Based on Similarity Computation and Learning-to-Rank

This repository provides the reference implementations of SIGEM* and LINOW*, improved variants of [SIGEM](https://github.com/mrhhyu/SIGEM/tree/main) and [LINOW](https://github.com/mrhhyu/SIGEM/tree/main), respectively.



LINOW* addresses the zero-similarity issue of LINOW in similarity computation for directed graphs, by considering both in-neighbors and out-neighbors while preserving link directions.

SIGEM* employs LINOW* in the embedding process; it inherits all the properties of SIGEM while improving its effectiveness in graph embedding.

## Installation and Usage
In order to run SIGEM, the following packages are required:
```
Python       = 3.12
tensorflow   = 2.18.0
numpy        = 1.26.4
scipy        = 1.13.1
scikit-learn = 1.5.2
tqdm         = 4.66.6
```
SIGEM_STAR can be run directly from the command line or migrated to your favorite IDE.
## Graph Format
1) A graph must be represented as a text file under the **edge list format** in which, each line corresponds to an edge in the graph, tab is used as the separator of the two nodes, and the node index is started from 0.
2) A single original link in an **undirected graph and its train/test splitted graphs** must be represented via two links in both directions.

## Running SIGEM*
SIGEM* has the following parameters: 
```
--graph: Input graph; default is ''

--dataset_name: Dataset name; default is None

--result_dir: Destination to save the embedding result, default is "output/" in the root directory

--dim: The embedding dimension, default is 128

--itr: Number of Iterations to compute LINOW, default is 5

--damping_factor: Damping factor for similarity computation, default is 0.2

--scaling_factor: Scaling factor to select top nodes (default is None); set it as 10 for link prediction and node classification tasks, and as 2 for the graph reconstruction task

--gpu: The flag indicating to run SIGEM on GPU, default is True

--bch_cpu: Batch size for computation on CPU; default is 128. It is ignored if --gpu=True

--bch_gpu: Batch size for computation on GPU; default is 128. It is ignored if --gpu=False

--prl_num: Number of parallel computation on CPU; default is 8. It is ignored if --gpu=True

--epc: Number of Epochs for training, default is 100

--lr: Learning rate (default is None); set it as 0.0030 for small graphs and as 0.0012 for very large graphs

--early_stop: Flag indicating to stop the training process if the loss stops improving, default is True

--wait_thr: Number of epochs with no loss improvement after which the training will be stopped, default is 10

--read_topK_nodes: Flag to read top nodes from a binary file, they will be selected by computing similarity otherwise; default is False

--topK_file: Path to the saved top nodes

--write_topK_nodes: Flag to write top nodes into a binary file for later usage; default is False

--topK_save_path: Path to write the top nodes into a binary file

--lt: Link type for similarity computation and graph embedding (default is 'both-links').

--ld: In-links traversal probability for LINOW*
```
### NOTE:
**1- When "--lt" is set to 'in-links', 'out-links', and 'both-links', the embedding process employs SIGEM based on in-links, SIGEM based on out-links, and SIGEM***.

**2- It is highly recommended** to set the values of (1) "--itr" and "--damping_factor" to 5 and 0.2, respectively, for all tasks, (2) "--scaling_factor" to 10, 10, and 2 for link prediction, node classification, and graph reconstruction tasks, respectively, (3) --ld to 0.9 for all tasks, and "--bch_cpu", "--bch_gpu", and "--lr" parameters to the ones explained in Section 5.3.1 of the paper.

## Citation:
> Masoud Reyhani Hamedani, Chang Yeop Jung, Seong-Un Cho, Jeong-Seok Oh, Seong Soo Oh, and Sang-Wook Kim. 2026. Simple is Better: A Simple yet Effective Graph Embedding Method Based on Similarity Computation and Learning-to-Rank. Accepted to be published in Computer Science and Information Systems Journal (ComSIS)
> 
> Masoud Reyhani Hamedani, Jeong-Seok Oh, Seong-Un Cho, and Sang-Wook Kim. 2025. SIGEM: A Simple yet Effective Similarity based Graph Embedding Method. In Proceedings of the 31st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD'25, August 03-07, 2025, Toronto, ON, Canada, pages: 2420-2431, https://doi.org/10.1145/3711896.3737128



