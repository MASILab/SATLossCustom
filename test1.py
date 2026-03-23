import numpy as np
from utils.losses import PDMatchingLoss
import torch 


class dummy():
    precal_PD = False

H, W = 64, 64

gt, pred_connected, pred_disjoint = np.zeros((1, 1, H, W), dtype=float), np.zeros((1, 1, H, W), dtype=float), np.zeros((1, 1, H, W), dtype=float)
gt[0][0][16:32][12:24] = 1
pred_connected[0][0][16:32][12:24] = 1
pred_disjoint[0][0][16:22][12:18] = 1
pred_disjoint[0][0][28:32][22:24] = 1

gt_t = torch.tensor(gt, dtype=float)
predc_t = torch.tensor(pred_connected, dtype=float)
predd_t = torch.tensor(pred_disjoint, dtype=float)

loss_fn = PDMatchingLoss(dummy())

loss_same = loss_fn(gt_t, predc_t)
loss_cut = loss_fn(gt_t, predd_t)

print(loss_same)
print(loss_cut)