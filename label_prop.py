from abc import abstractmethod
import torch
import dgl
import time
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import utils_data

import matplotlib.pyplot as plt
import torch.nn.functional as F
from data_gen import Generator
from dgl import DGLGraph
from scipy.sparse import coo_matrix
# from sklearn.metrics import classification_report
# from torch_geometric.datasets import TUDataset
# from torch_geometric.datasets import Planetoid
# from torch_geometric.data import DataLoader
# from torch_geometric.utils.convert import to_networkx
# import torch_geometric.data
# import torch_geometric.utils

class BaseLabelPropagation:
    """Abstract class for Label Propagation:
    can be extended with new propagation rules"""
    # Parameters: adj_matrix: torch.FloatTensor Adjacency matrix of the graph.

    def __init__(self, adj_matrix):
        self.norm_adj_matrix = self._normalize(adj_matrix)
        self.n_nodes = adj_matrix.size(0)
        self.one_hot_labels = None
        self.n_classes = None
        self.labeled_mask = None
        self.unlabeled_mask = None
        self.predictions = None
        self.accuracy = None
        self.true_labeling = None
        self.idx_test = None

    @staticmethod
    @abstractmethod
    def _normalize(adj_matrix):
        raise NotImplementedError("_normalize must be implemented")

    @abstractmethod
    def _propagate(self):
        raise NotImplementedError("_propagate must be implemented")

    @abstractmethod
    def _accuracy(self, output, labels):
        raise NotImplementedError("_propagate must be implemented")

    def _one_hot_encode(self, labels):
        # Get the number of classes
        classes = torch.unique(labels)
        classes = classes[classes != -1]
        self.n_classes = classes.size(0)
        # One-hot encode labeled data instances and zero rows corresponding to unlabeled instances
        unlabeled_mask = (labels == -1)
        labels = labels.clone()  # defensive copying
        labels[unlabeled_mask] = 0
        self.one_hot_labels = torch.zeros((self.n_nodes, self.n_classes), dtype=torch.float)
        self.one_hot_labels = self.one_hot_labels.scatter(1, labels.unsqueeze(1), 1)
        self.one_hot_labels[unlabeled_mask, 0] = 0
        # self.labeled_mask = ~unlabeled_mask

    def encode_onehot(self,labels):
        classes = set(labels)
        self.n_classes = len(classes)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        self.one_hot_labels = labels_onehot

    def fit(self, labels, max_iter, tol):
        "Label Propagation Setup"
        # Parameters labels, Maximum number of iterations allowed, Convergence tolerance"
        self._one_hot_encode(labels)
        self.predictions = self.one_hot_labels.clone()
        prev_predictions = torch.zeros((self.n_nodes, self.n_classes), dtype=torch.float)

        dur = []
        results = []
        "Accuracy calculation"
        ratio = self._accuracy(labels[self.unlabeled_mask], true_labeling[self.unlabeled_mask])
        results.append(ratio)

        for i in range(1, max_iter+1 ):
            # Stop iterations if the output is considered at a steady state
            variation = torch.abs(self.predictions - prev_predictions).sum().item()
            if variation < tol:
                print(f"The method stopped after {i} iterations. The variation at the last iteration = {variation:.4f}.")

                break

            prev_predictions = self.predictions
            # Can be used for observing the predictions after iterations
            prev_propagation_output_labels = self.predict_classes()

            t = time.time()

            "Propagate labels"
            self._propagate()

            ''' Accuracy calculation'''
            label_propagation_output_labels = self.predict_classes()
            ratio = self._accuracy(label_propagation_output_labels[self.unlabeled_mask], true_labeling[self.unlabeled_mask])

            if i < 101:
                dur_i = time.time() - t
                print(f"The accuracy after the {i}th iteration is {ratio}. Time required for the iteration: {time.time() - t}")

                results.append(ratio)
                dur.append(dur_i)

            ''' Accuracy calculation'''

        "Save the accuracy results and duration"
        np.save('LP/LP1', results)
        print("Saved")
        # np.save('LP_dur2', dur)

    def predict(self):
        return self.predictions

    def predict_classes(self):
        # print(self.predictions)
        # print(self.predictions.max(dim=1).indices)
        return self.predictions.max(dim=1).indices


class LabelPropagation(BaseLabelPropagation):
    def __init__(self, adj_matrix):
        super().__init__(adj_matrix)

    @staticmethod
    def _normalize(adjacency_matrix):
        "Computes D^-1 * A"
        # Parameters: A : Adjacency matrix

        degree_matrix = adjacency_matrix.sum(dim=1)
        degree_matrix[degree_matrix == 0] = 1
        return adjacency_matrix / degree_matrix[:, None]

    def _propagate(self):
        self.predictions = torch.matmul(self.norm_adj_matrix, self.predictions)

        # Put back already known labels
        self.predictions[self.labeled_mask] = self.one_hot_labels[self.labeled_mask]

    def _accuracy(self, output, labels):
        preds = torch.LongTensor(labels)
        # preds = output.max(1)[1].type_as(labels)
        correct = output.eq(preds).double()
        correct = correct.sum()
        return correct / len(labels)

    def fit(self, labels, max_iter=1000, tol=1e-7):
        super().fit(labels, max_iter, tol)

def accuracy(output, labels):
    preds = labels
    correct = output.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def generateSBM():
    "Stochastic Block Model"
    "Class sizes of the SBM graph"
    number_of_nodes = 1000
    sizes = [250, 250, 250, 250]
    "Symmetric Block Matrix of SBM"
    # probs = [[0.6, 0.4, 0.4, 0.4],
    #          [0.4, 0.6, 0.4, 0.4],
    #          [0.4, 0.4, 0.6, 0.4],
    #          [0.4, 0.4, 0.4, 0.6]]
    probs = [[0.5, 0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5]]
    graph = nx.stochastic_block_model(sizes, probs, seed=1)
    return graph

def loadRealWorldDataset(dataset_name):
    g, features, labels, num_features, num_labels = utils_data.load_new_data(dataset_name)
    true_labeling = labels
    g = dgl.to_networkx(g)
    return true_labeling, g, labels

"Real-World Dataset Loader"
"Enter the name of the dataset"
dataset_name = "film"
# possible inputs: "texas", "chameleon", "cornell", "film", "squirrel", "wisconsin"
loadRealWorldDataset(dataset_name)
true_labeling, g, labels = loadRealWorldDataset(dataset_name)
node_count = len(labels)

"For loading Citation Networks: Cora, Citeseer, Pubmed"
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
# dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
# cora = dataset[0]
# g = to_networkx(cora)
# true_labeling = cora.y[list(g.nodes)].numpy()
# labels = np.full(g.number_of_nodes(), -1)

"SBM Graph Loader"
# g = generateSBM()
# # Generate Random Initial Labeling
# gen = Generator()
# w_med, labels_gen = gen.SBM_multiclass(0.9, 0.2, 1000, 4)
# node_count = len(g.nodes)
# labels = labels_gen
# # Generate Unlabeled Instances
# # labels = np.full(node_count, -1.)
#
# "Ground Truth Labels"
# true_labeling = np.full(node_count, -1.)
# for x in range(0, 250):
#     true_labeling[x] = 0
# for x in range(250, 500):
#     true_labeling[x] = 1
# for x in range(500, 750):
#     true_labeling[x] = 2
# for x in range(750, 1000):
#     true_labeling[x] = 3



"Labeled Data"
# Ratio of Labeled Data / Data
given_labels_ratio = 0.15
X_unlabelled, X_given_labels, Y_unlabelled, Y_given_labels = train_test_split(range(node_count), range(node_count), shuffle=True, train_size=(1-given_labels_ratio),
                                                        test_size=given_labels_ratio)

# Generate Unlabeled Instances or use random generator
labels = np.full(node_count, -1.)

for x in Y_given_labels:
    labels[x] = true_labeling[x]

raw_data = labels
# g = to_networkx(data)

# Create input tensors
adj_matrix = nx.adjacency_matrix(g).toarray()
adj_matrix_t = torch.FloatTensor(adj_matrix)

"If self-loops wanted"
# adj_matrix_t.fill_diagonal_(1.0)

# g = DGLGraph(g)
# adj_matrix_t = g.adjacency_matrix()

labels_t = torch.LongTensor(labels)

"Labeled Data Mask"
labelled_mask_t = torch.BoolTensor(node_count)
labelled_mask_t.fill_(False)
labelled_mask_t[Y_given_labels] = True
"Labeled and Unlabeled Data Mask"
unlabelled_mask_t = torch.BoolTensor(node_count)
unlabelled_mask_t.fill_(True)
unlabelled_mask_t[Y_given_labels] = False

# Label Propagation
print("Label Propagation")
label_propagation = LabelPropagation(adj_matrix_t)
label_propagation.n_classes = 4 #dataset.num_classes
label_propagation.true_labeling = true_labeling
label_propagation.labeled_mask = labelled_mask_t
label_propagation.unlabeled_mask = unlabelled_mask_t
label_propagation.fit(labels_t)
label_propagation_output_labels = label_propagation.predict_classes()
"End accuracy"
# acc_test = accuracy(label_propagation_output_labels[idx_test], true_labeling1[idx_test])

"Plot graphs"
# Idel for small graphs
# color_map = {-1: "gray", 0: "blue", 1: "green", 2: "red", 3: "cyan", 4: "black"}
#
# input_labels_colors = [color_map[l] for l in labels]
# lprop_labels_colors = [color_map[l] for l in label_propagation_output_labels.numpy()]
#
# plt.figure(figsize=(10, 6))
# ax1 = plt.subplot(1, 5, 1)
# ax2 = plt.subplot(1, 5, 2)
#
# ax1.title.set_text("Initial Data")
# ax2.title.set_text("Label Propagation")
#
# # pos = nx.spring_layout(g)
#
# plt.figure(1,figsize=(14,12))
# nx.draw(g, ax=ax1, cmap=plt.get_cmap('Set1'), node_color = input_labels_colors , node_size=10, linewidths=6)
# nx.draw(g, ax=ax2, cmap=plt.get_cmap('Set1'), node_color = label_propagation_output_labels.numpy(),node_size=10,linewidths=6)
# plt.show()

# Legend
# ax4 = plt.subplot(1, 5, 5)
# ax4.axis("off")
# legend_colors = ["black", "blue", "green", "red", "cyan"]
# legend_labels = ["unlabeled", "label 1", "label 2", "label 3", "label 4"]
# dummy_legend = [ax4.plot([], [], ls='-', c=c)[0] for c in legend_colors]
# plt.legend(dummy_legend, legend_labels)

# plt.savefig('Results.png', bbox_inches='tight')
