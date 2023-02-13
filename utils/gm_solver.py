import math

import torch
import torch.nn as nn
import numpy as np

from utils.fit import get_reg, get_coef


def get_aff_score(am, mat, beta=1):
    mat_vec = mat.transpose(0, 1).reshape(-1, 1)
    return torch.matmul(torch.matmul(mat_vec.transpose(0, 1), am), mat_vec)


def get_aff_score_norm(am, mat, ri):
    mat_vec = mat.transpose(0, 1).reshape(-1, 1)
    reg = get_reg(torch.sum(mat_vec), ri)
    return torch.matmul(torch.matmul(mat_vec.transpose(0, 1), am), mat_vec) * reg


def get_norm_affinity(affinity_matrix, n_src, n_tgt, mat_init):
    if len(affinity_matrix.shape) == 4:
        am = affinity_matrix.transpose(0, 1).transpose(2, 3).reshape(n_src * n_tgt, n_src * n_tgt)
    else:
        am = affinity_matrix

    mat = mat_init[:n_src, :n_tgt]
    le = int(max(np.sum(mat.cpu().detach().numpy()) - 2, 0))
    ri = int(min(np.sum(mat.cpu().detach().numpy()) + 2, min(n_tgt, n_src)))
    # ri = min(n_src, n_tgt)
    # le = max(1, math.floor(ri / 2.0))
    if ri - le > 1:
        alpha, beta, c = get_coef(list(range(le, ri + 1)))
    elif ri - le == 1:
        alpha, beta, c = get_coef([le, le + 0.5, ri])
    else:
        alpha, beta, c = get_coef([le - 0.5, le, le + 0.5])

    aff_score = get_aff_score(am, mat)
    am_alpha = torch.ones(am.shape).cuda() * aff_score
    am_beta = torch.eye(n_src * n_tgt).cuda() * aff_score
    am_new = am - alpha * am_alpha - beta * am_beta

    return am_new
