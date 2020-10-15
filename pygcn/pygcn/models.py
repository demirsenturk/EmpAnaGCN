import torch.nn as nn
import torch
import torch.nn.functional as F
from pygcn.layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        "Multi-Layered Structure of model"
        " New layers can be added here"
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        "The layers should be connected"
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        "Activation functions: ReLU: F.relu, Tanh: torch.tanh, Sigmoid: torch.sigmoid"
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)