# Libraries
import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from torch_geometric.data import DataLoader
import random
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# dataset import and feature generation
ds = pd.read_hdf("../Processed_Data/GSE92742_fully_restricted_prostate.hdf")

# X 
X = np.asarray(ds)
print("Shape of X: ", X.shape)

# indices 
indices = []
for i in range(len(X)):
	indices.append(i)

# y
y_p = pd.DataFrame(ds.index)
y_p = np.asarray(y_p)
y = []
for i in y_p:
	y.append(np.asarray(i[0])[5])
# print(y)

# One Hot Encoding
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
labels = np_utils.to_categorical(encoded_Y)
print("The classes in y are: " + str(encoder.classes_))
num_classes = len(encoder.classes_)
# Adjacency Matric Creation
adj = euclidean_distances(X, X)
print("Shape of Adjacency matrix: ", adj.shape)

# Data
X_train, X_test, y_train, y_test, train_idx, test_idx, A_train, A_test = train_test_split(X, y, indices, adj, test_size=0.2, random_state=42)

use_gpu = torch.cuda.is_available()

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if use_gpu:
    torch.cuda.manual_seed(42)

model, optimizer = None, None
best_acc = 0

# Model
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, init):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nclass, init=init)
        self.dropout = dropout

    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x, adj):
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)

model = GCN(
            nfeat = features.shape[1],
            nhid = opt.num_hidden,
            nclass = labels.max().item() + 1,
            dropout = opt.dropout,
            init = opt.init_type
    )


if use_gpu:
    model.cuda()
    features, adj, labels, idx_train, idx_val, idx_test = \
        list(map(lambda x: x.cuda(), [X, adj, labels, idx_train, idx_val, idx_test]))

features, adj, labels = list(map(lambda x : Variable(x), [X, adj, labels]))


