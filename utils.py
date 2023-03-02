import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_A(A, symmetry=True):
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d1 = torch.sum(A, 0)
        d1 = 1 / torch.sqrt(d1 + 1e-10)
        D1 = torch.diag_embed(d1)
        d2 = torch.sum(A, 1)
        d2 = 1 / torch.sqrt(d2 + 1e-10)
        D2 = torch.diag_embed(d2)
        L = torch.matmul(torch.matmul(D1, A), D2)
    return L


def generate_adj(A, K,flag=1):
    #A的次方
    if flag==1:
        support = []
        for i in range(K):
            if i == 0:
                support.append(torch.eye(A.shape[-1]).cuda())
            elif i == 1:
                support.append(A)
            else:
                temp = torch.matmul(support[-1], A)
                support.append(temp)
    #chebynet
    else:
        support = []
        for i in range(K):
            if i == 0:
                support.append(torch.eye(A.shape[-1]).cuda())
            elif i == 1:
                support.append(A)
            else:
                temp = torch.matmul(2*A,support[-1],)-support[-2]
                support.append(temp)
    return support
