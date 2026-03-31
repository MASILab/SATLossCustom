import os
import re
from pathlib import Path
import nibabel as nib
import numpy as np
import torch

def natural_key(str1):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', str1)]

class Topo_dataloader(torch.utils.data.Dataset):
    def __init__(self, root, mode):
        if mode == 'train':
            self.img_root = os.path.join(root, 'train')
            self.gt_root = os.path.join(root, 'train_labels')
        elif mode == 'val':
            self.img_root = os.path.join(root, 'val')
            self.gt_root = os.path.join(root, 'val_labels')
        elif mode == 'test':
            self.img_root = os.path.join(root, 'test')
            self.gt_root = os.path.join(root, 'test_labels')
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.img_list = [
            f for f in sorted(os.listdir(self.img_root), key=natural_key)
            if not f.startswith('.')
        ]
        self.gt_list = [
            f for f in sorted(os.listdir(self.gt_root), key=natural_key)
            if not f.startswith('.')
        ]
        assert len(self.img_list) == len(self.gt_list)
        print(f'{len(self.img_list)} {mode} images')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_name = self.img_list[item]
        gt_name = self.gt_list[item]
        img_path = os.path.join(self.img_root, img_name)
        gt_path = os.path.join(self.gt_root, gt_name)

        img_nii = nib.load(img_path)
        gt_nii = nib.load(gt_path)

        img = img_nii.get_fdata().astype(np.float32)
        gt = gt_nii.get_fdata().astype(np.float32)
        gt = (gt > 0).astype(np.float32)

        img = np.expand_dims(img, axis=0)  
        gt = np.expand_dims(gt, axis=0)

        img_t = torch.from_numpy(img).unsqueeze(0)
        gt_t = torch.from_numpy(gt).unsqueeze(0)

        img_t = img_t.squeeze(0)
        gt_t = gt_t.squeeze(0)

        return img_t, gt_t, img_name