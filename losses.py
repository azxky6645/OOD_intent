import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses
from sklearn.preprocessing import label_binarize


def binarize(T, n_classes):
    T = T.cpu().numpy()
    T = label_binarize(T, classes=range(0, n_classes))
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


def acos(x):
    eps = 1e-7
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return torch.acos(x)


def calculate_sin_from_cos(cos_val):
    sin_val = math.sqrt(1 - cos_val**2)
    return sin_val


# def cos_sim(i, j):
#     return F.linear(l2_norm(i), l2_norm(j))


def cos_sim(i, j):
    i = l2_norm(i)
    j = l2_norm(j)
    tensor1_2d = j.unsqueeze(0)
    tensor2_2d = i.unsqueeze(1)
    tensor2_2d = tensor2_2d.expand(-1, j.shape[0], -1)
    return torch.clamp(F.cosine_similarity(tensor1_2d, tensor2_2d, dim=2), -1, 1)


class Proxy_Anchor(nn.Module):
    def __init__(self, label_emb, n_classes, hidden_size, alpha=64): #n_classes: known + unknown
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        if label_emb == None:
            self.proxies = torch.nn.Parameter(torch.randn(n_classes, hidden_size).cuda())
            nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        else:
            self.proxies = torch.nn.Parameter(label_emb)

        self.n_classes = n_classes
        self.hidden_size = hidden_size

        self.mrg = torch.tensor(self.proxies)
        self.mrg = torch.nn.Parameter(self.mrg).cuda()

        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies
        cos = cos_sim(X, P)  # Calcluate cosine similarity
        all_mrg_P_cos = cos_sim(self.mrg, P)
        X_mrg_cos = cos_sim(X, self.mrg)
        mrg_P_cos = torch.diag(all_mrg_P_cos)

        P_one_hot = binarize(T=T, n_classes=self.n_classes)
        P_one_hot[:, -1] = 1
        N_one_hot = 1 - P_one_hot

        inner_mask = cos > mrg_P_cos

        pos_cos = torch.where(inner_mask == 1, X_mrg_cos, (cos - mrg_P_cos))
        neg_cos = torch.where(inner_mask == 1, (cos - mrg_P_cos), X_mrg_cos)

        pos_exp = torch.exp(-self.alpha * pos_cos)
        neg_exp = torch.exp(self.alpha * neg_cos)

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.n_classes

        loss = pos_term + neg_term
        return loss, P, self.mrg, mrg_P_cos, pos_cos, neg_cos

