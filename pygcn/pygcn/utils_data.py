#  MIT License
#  Copyright (c) 2019 Geom-GCN Authors

import os
import re

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit

import utils1

def load_new_data(dataset_name, splits_file_path=None, train_percentage=0.2, val_percentage=0.2, embedding_mode=None,
              embedding_method=None,
              embedding_method_graph=None, embedding_method_space=None):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, _, _, _ = utils1.load_data(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.Graph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                                f'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    features = utils1.preprocess_features(features)

    if not embedding_mode:
        g = DGLGraph(adj + sp.eye(adj.shape[0]))
    else:
        if embedding_mode == 'ExperimentTwoAll':
            embedding_file_path = os.path.join('embedding_method_combinations_all',
                                               f'outf_nodes_relation_{dataset_name}all_embedding_methods.txt')
        elif embedding_mode == 'ExperimentTwoPairs':
            embedding_file_path = os.path.join('embedding_method_combinations_in_pairs',
                                               f'outf_nodes_relation_{dataset_name}_graph_{embedding_method_graph}_space_{embedding_method_space}.txt')
        else:
            embedding_file_path = os.path.join('structural_neighborhood',
                                           f'outf_nodes_space_relation_{dataset_name}_{embedding_method}.txt')
        space_and_relation_type_to_idx_dict = {}

        with open(embedding_file_path) as embedding_file:
            for line in embedding_file:
                if line.rstrip() == 'node1,node2	space	relation_type':
                    continue
                line = re.split(r'[\t,]', line.rstrip())
                assert (len(line) == 4)
                assert (int(line[0]) in G and int(line[1]) in G)
                if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                    space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                        space_and_relation_type_to_idx_dict)
                if G.has_edge(int(line[0]), int(line[1])):
                    G.remove_edge(int(line[0]), int(line[1]))
                G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                    (line[2], int(line[3]))])

        space_and_relation_type_to_idx_dict['self_loop'] = len(space_and_relation_type_to_idx_dict)
        for node in sorted(G.nodes()):
            if G.has_edge(node, node):
                G.remove_edge(node, node)
            G.add_edge(node, node, subgraph_idx=space_and_relation_type_to_idx_dict['self_loop'])
        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        g = DGLGraph(adj)

        for u, v, feature in G.edges(data='subgraph_idx'):
            g.edges[g.edge_id(u, v)].data['subgraph_idx'] = th.tensor([feature])

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)

    # print(features)
    # print(g.adjacency_matrix())

    # # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
    # degs = g.in_degrees().float()
    # norm = th.pow(degs, -0.5).cuda()
    # norm[th.isinf(norm)] = 0
    # g.ndata['norm'] = norm.unsqueeze(1)

    return g, features, labels, num_features, num_labels
