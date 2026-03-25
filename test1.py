import nibabel as nib
import numpy as np
import torch
from pathlib import Path

from utils.losses import PDMatchingLoss


class Dummy:
    precal_PD = False


path = Path('/home/local/VANDERBILT/shij18/eye_group_tem/modified_seg/PHOTON-x-13853-x-13853_20161028_MR-x-9-SEG.nii.gz')

nii = nib.load(path)
data = nii.get_fdata()

data = (data > 0).astype(np.float32)

vol = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

loss_fn = PDMatchingLoss(Dummy())

loss = loss_fn(vol, vol)

print("self-loss:", loss.item())