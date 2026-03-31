import os
from os.path import exists
import torch
import numpy as np
import nibabel as nib
from utils.dataset import Topo_dataloader
from utils.metrics import pixel_accuracy_item, dice_score_item, clDice, precision, recall, f1score


def tester(args, model):
    accmeter = []
    dicemeter = []
    cldicemeter = []
    premeter = []
    recmeter = []
    f1meter = []

    test_dataset = Topo_dataloader(os.path.join(args.dataroot, args.dataset), 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    save_pred_dir = os.path.join(args.exp_output_dir, 'pred')
    save_pred_g_dir = os.path.join(args.exp_output_dir, 'pred_g')

    if not exists(save_pred_dir):
        os.makedirs(save_pred_dir)
    if not exists(save_pred_g_dir):
        os.makedirs(save_pred_g_dir)

    model = model.eval()

    with torch.no_grad():
        for step, (samples, target, img_names) in enumerate(test_loader):
            samples = samples.to(torch.float32).to(args.device)
            target = target.to(torch.float32).to(args.device)

            # forward — model outputs logits (N, 1, D, H, W)
            pred_logits = model(samples)
            pred_prob = torch.sigmoid(pred_logits)
            pred_binary = (pred_logits > 0.0).to(torch.float32)

            # metrics
            acc = pixel_accuracy_item(pred_binary, target)
            dice = dice_score_item(pred_binary, target)
            cldice = clDice(pred_binary.cpu(), target.cpu())
            prec = precision(pred_binary, target)
            recl = recall(pred_binary, target)
            f1 = f1score(pred_binary, target)

            accmeter.extend(acc)
            dicemeter.extend(dice)
            cldicemeter.extend(cldice)
            premeter.extend(prec)
            recmeter.extend(recl)
            f1meter.extend(f1)

            # save predictions as NIfTI
            N = pred_binary.size(0)
            for b_idx in range(N):
                save_name = img_names[b_idx].replace('.nii.gz', '')

                # binary prediction (N, 1, D, H, W) -> (D, H, W)
                pred_bin_np = pred_binary[b_idx, 0].cpu().numpy().astype(np.uint8)
                pred_prob_np = pred_prob[b_idx, 0].cpu().numpy().astype(np.float32)

                metrics_str = '{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(
                    acc[b_idx], dice[b_idx], cldice[b_idx],
                    prec[b_idx], recl[b_idx], f1[b_idx]
                )

                bin_filename = '{}__{}.nii.gz'.format(save_name, metrics_str)
                prob_filename = '{}__{}.nii.gz'.format(save_name + '_prob', metrics_str)

                nib.save(
                    nib.Nifti1Image(pred_bin_np, np.eye(4)),
                    os.path.join(save_pred_dir, bin_filename)
                )
                nib.save(
                    nib.Nifti1Image(pred_prob_np, np.eye(4)),
                    os.path.join(save_pred_g_dir, prob_filename)
                )

                print('[{}/{}] {} | Acc:{:.4f} Dice:{:.4f} clDice:{:.4f} Prec:{:.4f} Rec:{:.4f} F1:{:.4f}'.format(
                    step + 1, len(test_loader), img_names[b_idx],
                    acc[b_idx], dice[b_idx], cldice[b_idx],
                    prec[b_idx], recl[b_idx], f1[b_idx]
                ))

    acc_mean = np.mean(accmeter)
    dice_mean = np.mean(dicemeter)
    cldicemean = np.mean(cldicemeter)
    precmean = np.mean(premeter)
    reclmean = np.mean(recmeter)
    f1mean = np.mean(f1meter)

    print('\n=== Test Results ===')
    print('Acc:{:.4f}, Dice:{:.4f}, clDice:{:.4f}, Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}'.format(
        acc_mean, dice_mean, cldicemean, precmean, reclmean, f1mean
    ))