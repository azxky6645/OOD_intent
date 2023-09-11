import numpy as np
import torch
import logging
import losses
import json
from tqdm import tqdm
import torch.nn.functional as F
import math
from torchmetrics import Accuracy, F1Score
from losses import cos_sim


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


def pred_nearest_proxy(x_emb, proxy, device='cuda:0'):
    cos_sim = F.linear(x_emb, proxy)
    y_pred = cos_sim.topk(1)[1]
    y_pred = y_pred.squeeze(-1)
    return y_pred

# def pred_proxy_knn(x_emb, train_x_emb, train_y_emb, proxy, K, data_key ,device='cuda:0'):
#     proxy_label = torch.tensor(list(LABEL_DICT[data_key].values())).to(device)
#
#     train_x_emb_with_proxy = torch.cat([l2_norm(train_x_emb), proxy])
#     train_y_emb_with_proxy = torch.cat([train_y_emb, proxy_label])
#
#     cos_sim = F.linear(x_emb, train_x_emb_with_proxy)
#     y_pred = train_y_emb_with_proxy[cos_sim.topk(K)[1]]
#     y_pred = torch.mode(y_pred, dim=1)[0]
#     return y_pred
#
#
# def pred_classwise_avg(x_emb, train_x_emb, train_y_emb, proxy, data_key, device='cuda:0'):
#     proxy_label = torch.tensor(list(LABEL_DICT[data_key].values())).to(device)
#
#     train_x_emb_with_proxy = torch.cat([l2_norm(train_x_emb), proxy])
#     train_y_emb_with_proxy = torch.cat([train_y_emb, proxy_label])
#
#     cos_sim = F.linear(x_emb, train_x_emb_with_proxy)
#
#     y_pred_list = []
#
#     for c in proxy_label:
#         class_mask = torch.logical_not(torch.eq(train_y_emb_with_proxy, c))
#         class_masked_sim = torch.masked_fill(cos_sim, class_mask, 0)
#         class_count = torch.count_nonzero(class_masked_sim, dim=1)
#         class_avg_sim = class_masked_sim.sum(dim=1) / class_count
#
#         y_pred_list.append(class_avg_sim)
#
#     y_pred_list = torch.stack(y_pred_list, dim=1)
#     y_pred = y_pred_list.max(dim=1)[1]
#
#     return y_pred


def proxy_distance_with_margin(x_emb, proxy, margin, old):
    if old == False:
        ood_proxy = torch.zeros(len(proxy)).cuda()    # If all points do not belong to the proxy
        ood_proxy[-1] = 1                       # like (0,0,0,...,0,1)
        X_P = cos_sim(x_emb, proxy)
        margin_proxy_sim = torch.diag(cos_sim(margin, proxy))
        # x_belong_proxy = X_P * (X_P > margin_proxy_sim)
        x_belong_proxy = torch.where(X_P > margin_proxy_sim, X_P, -2)
        # x_belong_proxy[:, -1] = -2
        zero_rows = torch.where(torch.all(x_belong_proxy == -2, dim=1))[0]
        x_belong_proxy[zero_rows] = ood_proxy
        y_pred = torch.argmax(x_belong_proxy, dim=1)    # size like (batch, 1)

    elif old == True:
        ood_proxy = torch.zeros(len(proxy)).cuda()  # If all points do not belong to the proxy
        ood_proxy[-1] = 1  # like (0,0,0,...,0,1)
        X_P = cos_sim(x_emb, proxy)
        margin_proxy_sim = margin
        # x_belong_proxy = X_P * (X_P > margin_proxy_sim)
        x_belong_proxy = torch.where(X_P > margin_proxy_sim, X_P, -2)
        # x_belong_proxy[:, -1] = -2
        zero_rows = torch.where(torch.all(x_belong_proxy == -2, dim=1))[0]
        x_belong_proxy[zero_rows] = ood_proxy
        y_pred = torch.argmax(x_belong_proxy, dim=1)  # size like (batch, 1)



    return y_pred
