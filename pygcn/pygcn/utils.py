import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import utils_data

from dgl import DGLGraph
from data_gen import Generator
from sklearn.model_selection import train_test_split
from dgl.data import citation_graph as citegrh

import dgl
from itertools import chain
# import torch_geometric
# import torch_geometric.utils

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_citation_network():
    "Citeseer dataset"
    data = citegrh.load_citeseer()

    "Pubmed Dataset"
    # data = citegrh.load_pubmed()

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, train_mask, test_mask

def load_real_world_dataset():
    """Load real-world dataset"""

    "Enter the name of the dataset"
    dataset_name = "texas"
    # possible inputs: "texas", "chameleon", "cornell", "film", "squirrel", "wisconsin"

    g, features, labels, num_features, num_labels = utils_data.load_new_data(dataset_name)
    print('Loading {} dataset...'.format(dataset_name))

    number_of_nodes = len(labels)

    X_train, X_test, Y_train, Y_test = train_test_split(range(number_of_nodes), range(number_of_nodes), shuffle=True, train_size=0.80,
                                                        test_size=0.20)
    "train/test/validation split"
    X_train, X_val, Y_train, Y_val = train_test_split(X_test, Y_test, shuffle=True, train_size=0.30,
                                                             test_size=0.70)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, shuffle=True, train_size=0.30,
                                                             test_size=0.70)
    idx_train = X_train
    idx_val = X_val
    idx_test = X_test

    " N x N identity matrix"
    # features = torch.eye(number_of_nodes,number_of_nodes)

    "Node features"
    features = torch.FloatTensor(features)
    adj = g.adjacency_matrix()
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_cora(path="../data/cora/", dataset="cora"):
    """For loading Cora dataset"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    "Train/Test/Validation Split Masks"
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_SBM():
    """Generate and load SBM graph"""

    "Class sizes of the SBM graph"
    number_of_nodes = 1000
    sizes = [250, 250, 250, 250]
    "Symmetric Block Matrix of SBM"
    probs = [[0.9, 0.2, 0.2, 0.2],
             [0.2, 0.9, 0.2, 0.2],
             [0.2, 0.2, 0.9, 0.2],
             [0.2, 0.2, 0.2, 0.9]]
    # probs = [[0.80, 0.30, 0.30, 0.30],
    #          [0.30, 0.80, 0.30, 0.30],
    #          [0.30, 0.30, 0.80, 0.30],
    #          [0.30, 0.30, 0.30, 0.80]]
    # probs = [[0.75, 0.35, 0.35, 0.35],
    #          [0.35, 0.75, 0.35, 0.35],
    #          [0.35, 0.35, 0.75, 0.35],
    #          [0.35, 0.35, 0.35, 0.75]]
    # probs = [[0.6, 0.4, 0.4, 0.4],
    #          [0.4, 0.6, 0.4, 0.4],
    #          [0.4, 0.4, 0.6, 0.4],
    #          [0.4, 0.4, 0.4, 0.6]]
    # probs = [[0.5, 0.5, 0.5, 0.5],
    #          [0.5, 0.5, 0.5, 0.5],
    #          [0.5, 0.5, 0.5, 0.5],
    #          [0.5, 0.5, 0.5, 0.5]]
    # probs = [[0.9, 0.01, 0.01, 0.01],
    #          [0.01, 0.9, 0.01, 0.01],
    #          [0.01, 0.01, 0.9, 0.01],
    #          [0.01, 0.01, 0.01, 0.9]]
    # probs = [[0.7, 0.25, 0.45, 0.45],
    #          [0.25, 0.5, 0.25, 0.3],
    #          [0.45, 0.25, 0.9, 0.2],
    #          [0.45, 0.3, 0.2, 0.6]]

    "SBM Graph Generation"
    g = nx.stochastic_block_model(sizes, probs, selfloops=True )

    true_labeling = np.full(number_of_nodes, -1.)
    "Ground Truth Labels"
    for x in range(0, 250):
        true_labeling[x] = 0
    for x in range(250, 500):
        true_labeling[x] = 1
    for x in range(500, 750):
        true_labeling[x] = 2
    for x in range(750, 1000):
        true_labeling[x] = 3
    true_labeling = torch.LongTensor(true_labeling)

    "Random Label Generator"
    # arguments: Intra-connection probability, Inter-connection probability, number of nodes, number of classes
    gen = Generator()
    w_med, labels_gen = gen.SBM_multiclass(0.9, 0.2, number_of_nodes, 4)
    labels = labels_gen

    # Ratio of Labeled Data / Data
    given_labels_ratio = 0.20

    "train/test/validation Split for complete Graph"
    # X_train_validation, X_test, Y_train_validation, Y_test = train_test_split(range(number_of_nodes), range(number_of_nodes), shuffle=True, train_size=0.80,
    #                                                     test_size=0.20)
    # X_train, X_validation, Y_train, Y_val = train_test_split(X_train_validation, Y_train_validation, shuffle=True, train_size=0.75,
    #                                                          test_size=0.25)
    # "Given Labels"
    # X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_train, Y_train, shuffle=True,
    #                                                                               train_size=(1 - given_labels_ratio),
    #                                                                               test_size=given_labels_ratio)
    # for x in Y_given_labels:
    #     labels[x] = true_labeling[x]
    # Y_train1 = Y_given_labels
    #
    # X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_validation, Y_val, shuffle=True,
    #                                                                               train_size=(1 - given_labels_ratio),
    #                                                                               test_size=given_labels_ratio)
    # for x in Y_given_labels:
    #     labels[x] = true_labeling[x]
    # Y_val1 = Y_given_labels
    #
    # X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_test, Y_test, shuffle=True,
    #                                                                               train_size=(1 - given_labels_ratio),
    #                                                                               test_size=given_labels_ratio)
    # for x in Y_given_labels:
    #     labels[x] = true_labeling[x]
    # Y_test1 = Y_given_labels

    ###########################################################

    "Give for each class the given nodes and training/test/validation split seperately "
    "Class 1  - Training/Test/Validation"
    X_train, X_test, Y_train, Y_test = train_test_split( range(250), range(250), shuffle=True, train_size=0.80, test_size =0.20)
    X_train, X_validation, Y_train, Y_val = train_test_split(X_train, Y_train, shuffle=True, train_size=0.75, test_size=0.25)

    X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_train, Y_train, shuffle=True,
                                                                                  train_size=(1-given_labels_ratio), test_size=given_labels_ratio)
    for x in Y_given_labels:
        labels[x] = true_labeling[x]
    Y_train1 = Y_given_labels

    X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_validation, Y_val, shuffle=True,
                                                                                  train_size=(1-given_labels_ratio), test_size=given_labels_ratio)
    for x in Y_given_labels:
        labels[x] = true_labeling[x]
    Y_val1 = Y_given_labels

    X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_test, Y_test, shuffle=True,
                                                                                  train_size=(1-given_labels_ratio), test_size=given_labels_ratio)
    for x in Y_given_labels:
        labels[x] = true_labeling[x]
    Y_test1 = Y_given_labels

    "Class 2  - Training/Test/Validation"
    X_train, X_test, Y_train, Y_test = train_test_split(range(250,500), range(250,500), shuffle=True, train_size=0.80,
                                                        test_size=0.20)
    X_train, X_validation, Y_train, Y_val = train_test_split(X_train, Y_train, shuffle=True, train_size=0.75,
                                                             test_size=0.25)

    X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_train, Y_train, shuffle=True,
                                                                                  train_size=(1-given_labels_ratio), test_size=given_labels_ratio)
    for x in Y_given_labels:
        labels[x] = true_labeling[x]
    Y_train1 = Y_train1 + Y_given_labels

    X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_validation, Y_val, shuffle=True,
                                                                                  train_size=(1-given_labels_ratio), test_size=given_labels_ratio)
    for x in Y_given_labels:
        labels[x] = true_labeling[x]
    Y_val1 = Y_val1 + Y_given_labels

    X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_test, Y_test, shuffle=True,
                                                                                  train_size=(1-given_labels_ratio), test_size=given_labels_ratio)
    for x in Y_given_labels:
        labels[x] = true_labeling[x]
    Y_test1 = Y_test1 + Y_given_labels

    "Class 3 - Training/Test/Validation"
    X_train, X_test, Y_train, Y_test = train_test_split(range(500,750), range(500,750), shuffle=True, train_size=0.80,
                                                        test_size=0.20)
    X_train, X_validation, Y_train, Y_val = train_test_split(X_train, Y_train, shuffle=True, train_size=0.75,
                                                             test_size=0.25)

    X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_train, Y_train, shuffle=True,
                                                                                  train_size=(1-given_labels_ratio), test_size=given_labels_ratio)
    for x in Y_given_labels:
        labels[x] = true_labeling[x]
    Y_train1 = Y_train1 + Y_given_labels

    X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_validation, Y_val, shuffle=True,
                                                                                  train_size=(1-given_labels_ratio), test_size=given_labels_ratio)
    for x in Y_given_labels:
        labels[x] = true_labeling[x]
    Y_val1 = Y_val1 + Y_given_labels

    X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_test, Y_test, shuffle=True,
                                                                                  train_size=(1-given_labels_ratio), test_size=given_labels_ratio)
    for x in Y_given_labels:
        labels[x] = true_labeling[x]
    Y_test1 = Y_test1 + Y_given_labels

    "Class 4 - Training/Test/Validation"
    X_train, X_test, Y_train, Y_test = train_test_split(range(750,1000), range(750,1000), shuffle=True, train_size=0.80,
                                                        test_size=0.20)
    X_train, X_validation, Y_train, Y_val = train_test_split(X_train, Y_train, shuffle=True, train_size=0.75,
                                                             test_size=0.25)

    X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_train, Y_train, shuffle=True,
                                                                                  train_size=(1-given_labels_ratio), test_size=given_labels_ratio)
    for x in Y_given_labels:
        labels[x] = true_labeling[x]
    Y_train1 = Y_train1 + Y_given_labels

    X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_validation, Y_val, shuffle=True,
                                                                                  train_size=(1-given_labels_ratio), test_size=given_labels_ratio)
    for x in Y_given_labels:
        labels[x] = true_labeling[x]
    Y_val1 =  Y_val1 + Y_given_labels

    X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(X_test, Y_test, shuffle=True,
                                                                                  train_size=(1-given_labels_ratio), test_size=given_labels_ratio)
    for x in Y_given_labels:
        labels[x] = true_labeling[x]
    Y_test1 = Y_test1 + Y_given_labels

    ###########################################################

    "NxN Identity Matrix"
    features = torch.eye(number_of_nodes, number_of_nodes)

    ###########################################################
    "Artificial Node Features"
    # synthetic_features = []
    # for x in range(1000):
    #     if true_labeling[x] == 0:
    #         synthetic_features.append(100)
    #     elif true_labeling[x] == 1:
    #         synthetic_features.append(200)
    #     elif true_labeling[x] == 2:
    #         synthetic_features.append(300)
    #     elif true_labeling[x] == 3:
    #         synthetic_features.append(400)
    # # Gaussian Noise
    # # arguments: mean μ, and variance σ^2
    # noise = np.random.normal(0, 0.3, len(synthetic_features))
    # # Noise added
    # synthetic_features = synthetic_features + noise
    # synthetic_features = encode_onehot(synthetic_features)
    # features = torch.FloatTensor(synthetic_features)
    ###########################################################

    "Labels of the Graph"
    labels = torch.LongTensor(labels)

    g = DGLGraph(g)
    "Adjacency Matrix of the Graph"
    adj = g.adjacency_matrix()

    "Train/Test/Validation Split Masks"
    idx_train = Y_train1
    idx_val = Y_val1
    idx_test = Y_test1

    return adj, features, labels, idx_train, idx_val, idx_test, true_labeling

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
