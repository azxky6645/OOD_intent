import torch
from torchmetrics import F1Score, Accuracy, Recall, Precision
from kmeans_pytorch import kmeans
from utils import l2_norm


def calc_accuracy(x, y):
    max_vals, max_indices = torch.max(x, 1)
    train_acc = (max_indices == y).sum().data.detach().cpu().numpy()/max_indices.size()[0]
    return train_acc
#
#
# def accuracy(num_classes, pred, target, device="cuda"):
#     acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
#     result = acc(preds=pred, target=target)
#     return result
#
#
# def f1_score(num_classes, pred, target, device="cuda"):
#     f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted').to(device)
#     result = f1(preds=pred, target=target)
#     return result
#
#
# def recall(num_classes, pred, target, device="cuda"):
#     re = Recall(num_classes=num_classes, average='weighted').to(device)
#     result = re(preds=pred, target=target)
#     return result
#
#
# def precision(num_classes, pred, target, device="cuda"):
#     pr = Precision(num_classes=num_classes, average='weighted').to(device)
#     result = pr(preds=pred, target=target)
#     return result
#
