import torch
import torch.nn.functional as F
import numpy as np
import math
from torch_topological.nn import CubicalComplex
from skimage.morphology import skeletonize

getPersistentInfo = CubicalComplex(dim=3)


def _pad_to_cube(pred, target):
    _, _, D, H, W = target.size()

    target_dim = max(D, H, W)

    d_margin = target_dim - D
    h_margin = target_dim - H
    w_margin = target_dim - W

    d_left = d_margin // 2
    d_right = d_margin - d_left

    h_left = h_margin // 2
    h_right = h_margin - h_left

    w_left = w_margin // 2
    w_right = w_margin - w_left

    paddings = (
        w_left, w_right,
        h_left, h_right,
        d_left, d_right
    )

    if pred is not None:
        pred = F.pad(pred, paddings, "constant", 0.0)
    if target is not None:
        target = F.pad(target, paddings, "constant", 0.0)

    return pred, target


def pixel_accuracy(pred, target):
    assert pred.size() == target.size()

    correct = torch.sum(pred == target)
    total = pred.numel()

    acc = correct / total
    return [acc.item()]


def dice_score(pred, target):
    assert pred.size() == target.size()

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = torch.sum(pred * target)
    segmentation = torch.sum(pred)
    ground_truth = torch.sum(target)

    dice = (2.0 * intersection) / (segmentation + ground_truth + 1e-6)
    return [dice.item()]


def pixel_accuracy_item(pred, target):
    assert pred.size() == target.size()
    N = pred.size(0)
    acc_book = []

    for i in range(N):
        correct = torch.sum(pred[i] == target[i])
        total = pred[i].numel()

        acc = correct / total
        acc_book.append(acc.item())

    return acc_book


def dice_score_item(pred, target):
    assert pred.size() == target.size()
    N = pred.size(0)
    dice_book = []

    for i in range(N):
        pred_i = pred[i].contiguous().view(-1)
        target_i = target[i].contiguous().view(-1)

        intersection = torch.sum(pred_i * target_i)
        segmentation = torch.sum(pred_i)
        ground_truth = torch.sum(target_i)

        dice = (2.0 * intersection) / (segmentation + ground_truth + 1e-6)
        dice_book.append(dice.item())

    return dice_book


def BettiError(pred, target):
    assert pred.size() == target.size()
    N, C, D, H, W = target.size()
    assert C == 1

    if D != H or H != W:
        pred, target = _pad_to_cube(pred, target)

    pred = 1 - pred
    target = 1 - target
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)

    p_pred = getPersistentInfo(pred)
    p_tgt = getPersistentInfo(target)

    B0E = []
    B1E = []
    B2E = []

    for b_idx in range(N):
        Betti_p_0 = len(p_pred[b_idx][0][0].diagram)
        Betti_p_1 = len(p_pred[b_idx][0][1].diagram)
        Betti_p_2 = len(p_pred[b_idx][0][2].diagram)

        Betti_t_0 = len(p_tgt[b_idx][0][0].diagram)
        Betti_t_1 = len(p_tgt[b_idx][0][1].diagram)
        Betti_t_2 = len(p_tgt[b_idx][0][2].diagram)

        B0E.append(np.abs(Betti_p_0 - Betti_t_0))
        B1E.append(np.abs(Betti_p_1 - Betti_t_1))
        B2E.append(np.abs(Betti_p_2 - Betti_t_2))

    return (B0E, B1E, B2E)


def cl_score(v, s):
    return np.sum(v * s) / np.sum(s)


def clDice_ins(v_p, v_l):
    if len(v_p.shape) == 2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    elif len(v_p.shape) == 3:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    else:
        raise ValueError(f"Unsupported input shape for clDice_ins: {v_p.shape}")

    if tprec == 0 and tsens == 0:
        return 0.0
    elif math.isnan(tsens) and math.isnan(tprec):
        return 1.0
    elif not math.isnan(tsens) and not math.isnan(tprec):
        return 2 * tprec * tsens / (tprec + tsens)
    else:
        return 0.0


def clDice(x, y):
    assert x.size() == y.size()
    assert x.size(1) == 1

    # x, y expected shape: (N, 1, D, H, W) for 3D
    # or (N, 1, H, W) for 2D
    x = x.squeeze(1)
    y = y.squeeze(1)

    x = x.cpu().numpy()
    y = y.cpu().numpy()

    result = []

    for i in range(x.shape[0]):
        score = clDice_ins(x[i], y[i])
        if np.isnan(score):
            score = 1.0
        result.append(score)

    return result


def precision(x, y):
    assert x.size() == y.size()
    N = x.size(0)
    pr = []

    for i in range(N):
        tp = (torch.mul(y[i], (x[i] == y[i]))).sum().item()
        fp = ((x[i] - y[i]) == 1).sum().item()
        if tp == 0 and fp == 0:
            pr.append(1.0)
        else:
            pr.append(tp / (tp + fp))

    return pr


def recall(x, y):
    assert x.size() == y.size()
    N = x.size(0)
    re = []

    for i in range(N):
        tp = (torch.mul(y[i], (x[i] == y[i]))).sum().item()
        fn = ((y[i] - x[i]) == 1).sum().item()
        if tp == 0 and fn == 0:
            re.append(1.0)
        else:
            re.append(tp / (tp + fn))

    return re


def f1score(x, y):
    assert x.size() == y.size()
    N = x.size(0)
    f1 = []

    for i in range(N):
        tp = (torch.mul(y[i], (x[i] == y[i]))).sum().item()
        fp = ((x[i] - y[i]) == 1).sum().item()
        fn = ((y[i] - x[i]) == 1).sum().item()

        if tp == 0 and fp == 0 and fn == 0:
            f1.append(1.0)
        else:
            f1.append((2 * tp) / (2 * tp + fp + fn))

    return f1