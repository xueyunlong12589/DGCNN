import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio


def normalize_A(A, mask=None, symmetry=False):
    if mask is not None:
        A = A * mask
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d2 = torch.sum(A, 1)
        d2 = 1 / torch.sqrt(d2 + 1e-10)
        D2 = torch.diag_embed(d2)
        L = torch.matmul(torch.matmul(D2, A), D2)

    return L


def generate_cheby_adj(A, K):
    support = []
    for i in range(K):
        if i == 0:
            # support.append(torch.eye(A.shape[1]).cuda())
            support.append(torch.eye(A.shape[1]).cuda())
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support
