import numpy as np
import torch
from utils.losses import PDMatchingLoss


class Dummy:
    precal_PD = False


D, H, W = 32, 64, 64

gt = np.zeros((1, 1, D, H, W), dtype=np.float32)
pred_1 = np.zeros((1, 1, D, H, W), dtype=np.float32)
pred_2 = np.zeros((1, 1, D, H, W), dtype=np.float32)
pred_3 = np.zeros((1, 1, D, H, W), dtype=np.float32)

gt[0, 0, 8:24, 16:32, 12:24] = 1.0

pred_1[0, 0, 8:24, 16:32, 12:24] = 1.0

pred_2[0, 0, 8:16, 16:24, 12:18] = 1.0
pred_2[0, 0, 18:24, 26:32, 20:24] = 1.0

pred_3[0, 0, 8:12, 16:20, 12:16] = 1.0
pred_3[0, 0, 14:18, 24:28, 20:24] = 1.0
pred_3[0, 0, 20:24, 28:32, 14:18] = 1.0

gt_t = torch.tensor(gt, dtype=torch.float32)
pred1_t = torch.tensor(pred_1, dtype=torch.float32)
pred2_t = torch.tensor(pred_2, dtype=torch.float32)
pred3_t = torch.tensor(pred_3, dtype=torch.float32)

loss_fn = PDMatchingLoss(Dummy())

loss_1 = loss_fn(pred1_t, gt_t)
loss_2 = loss_fn(pred2_t, gt_t)
loss_3 = loss_fn(pred3_t, gt_t)

print("1 component:", loss_1.item())
print("2 components:", loss_2.item())
print("3 components:", loss_3.item())