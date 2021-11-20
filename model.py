import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution,Linear
from utils import normalize_A, generate_cheby_adj


class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out, dropout):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2], num_out))

    def forward(self, x,L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result


class DGCNN(nn.Module):
    def __init__(self, xdim, k_adj, num_out, nclass=3):
        #xdim: (batch_size*num_nodes*num_nodes, num_features)
        #k_adj: The number of layers of graphconvolution
        #num_out: The feature dimension of the output
        super(GNN, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, num_out, dropout)
        self.BN1 = nn.BatchNorm1d(xdim[2])
        self.fc1 = Linear(xdim[1] * num_out, 64)
        self.fc2=Linear(64,nclass)
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())
        nn.init.xavier_normal_(self.A)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        result=self.fc2(result)
        return result
