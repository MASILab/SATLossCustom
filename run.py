import os
import sys
from os.path import exists
from datetime import datetime
import argparse
import torch
import time

from utils.logger import Logger
from model import UNet, UNet_small
from train import trainer
from test import tester


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='EYE_NIFTI', type=str)
parser.add_argument('--dataroot', default='./data', type=str)
parser.add_argument('--img_D', default=64, type=int)
parser.add_argument('--img_H', default=128, type=int)
parser.add_argument('--img_W', default=128, type=int)
parser.add_argument('--in_channels', default=1, type=int)

parser.add_argument('--seed', default=2024, type=int)
parser.add_argument('--exp', default='name_your_experiment', type=str, help='description of the current experiment')
parser.add_argument('--expmode', default='train', type=str, choices=['train', 'test'])

parser.add_argument('--model', default='UNet', type=str, choices=['UNet', 'UNet_small'])

parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--saved', type=str, default='')

parser.add_argument('--epoch', default=30, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--lr_decay_epoch', default=[10], type=list)
parser.add_argument('--lr_decay_rate', default=0.1, type=float)
parser.add_argument('--wd', default=1e-3, type=float)
parser.add_argument('--betas', default=[0.9, 0.999])
parser.add_argument('--eps', default=1e-8, type=float)
parser.add_argument('--tau', default=1e-2, type=float)
parser.add_argument('--alpha', default=1, type=float)
parser.add_argument('--save_model_interval', type=int, default=10)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--print_log', type=int, default=10)
parser.add_argument(
    '--test_model',
    type=str,
    default='best_dice',
    choices=['epoc50', 'best_acc', 'best_dice'],
    help='which checkpoint to use for testing?'
)

parser.add_argument('--tloss_w', type=float, default=1e-2, help='weight for the topology loss')
parser.add_argument('--precal_PD', type=bool, default=False, help='whether to precompute GT persistent diagrams')

parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--num_workers', type=int, default=0)


def main():
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    args.exp_output_dir = os.path.join('./exp', args.exp)
    if args.expmode == 'test':
        args.checkpoint = os.path.join('./exp', args.exp, args.test_model + '.pth')
        args.exp_output_dir = os.path.join('./exp', args.exp, args.test_model)

    if not exists(args.exp_output_dir):
        os.makedirs(args.exp_output_dir)

    if args.expmode == 'train':
        sys.stdout = Logger(
            os.path.join(args.exp_output_dir, f'log_{datetime.now()}.txt'),
            sys.stdout
        )
    elif args.expmode == 'test':
        sys.stdout = Logger(
            os.path.join(args.exp_output_dir, f'testlog_{datetime.now()}.txt'),
            sys.stdout
        )

    print(args)

    if torch.cuda.is_available():
        args.device = 'cuda:' + args.gpu_id
    else:
        args.device = 'cpu'
    print('Using device:', args.device)

    # NOTE:
    # This assumes UNet / UNet_small definitions are converted to 3D versions.
    # Hb and Wb are kept here only because your current model signature still expects them.
    # If you later rewrite model.py cleanly for 3D, you should simplify this constructor.
    if args.model == 'UNet':
        model = UNet(
            args.in_channels,
            1,
            Hb=max(1, args.img_H // (2 ** 4)),
            Wb=max(1, args.img_W // (2 ** 4))
        )
    elif args.model == 'UNet_small':
        model = UNet_small(
            args.in_channels,
            1,
            Hb=max(1, args.img_H // (2 ** 4)),
            Wb=max(1, args.img_W // (2 ** 4))
        )

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    model = model.to(args.device)

    start_time = time.time()

    if args.expmode == 'train':
        trainer(args, model)
    elif args.expmode == 'test':
        tester(args, model)

    end_time = time.time()
    running_time = end_time - start_time
    running_time_hours = running_time / 3600

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = 'CPU'

    if args.expmode == 'train':
        print(
            f"Ran for {running_time:.2f} seconds "
            f"({running_time_hours:.2f} hours) using {gpu_name} "
            f"on {args.dataset} dataset trained for {args.epoch} epochs."
        )
    elif args.expmode == 'test':
        print(
            f"Ran for {running_time:.2f} seconds "
            f"({running_time_hours:.2f} hours) using {gpu_name} "
            f"on {args.dataset} dataset."
        )


if __name__ == '__main__':
    main()