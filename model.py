import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
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
                result1 = self.gc1[i](x, adj[i])
            else:
                result1 += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result


class GNN(nn.Module):
    def __init__(self, xdim, k_adj, num_out, dropout, nclass=3):
        super(GNN, self).__init__()

        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, 32, dropout)
        #self.layer2 = IAG([xdim[0], xdim[1], 32], k_adj, 64, dropout)
        #self.layer3 = Chebynet([xdim[0], xdim[1], 64], k_adj, 128, dropout)
        self.dropout1 = nn.Dropout2d(dropout)
        #self.dropout2 = nn.Dropout2d(dropout)
        self.BN1 = nn.BatchNorm1d(5)
        self.fc1 = nn.Linear(xdim[1] * 32, 3)
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())
        nn.init.kaiming_normal_(self.A, mode='fan_in')

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(A)
        result = self.layer1(x, L)

        result = result.reshape(x.shape[0], -1)

        result = self.fc1(result)

        return result
