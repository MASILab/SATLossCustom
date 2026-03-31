import os
import torch
import numpy as np
import wandb

from utils.losses import PDMatchingLoss
from utils.dataset import Topo_dataloader
from utils.tools import adj_lr
from utils.metrics import pixel_accuracy, dice_score


def train_one_epoch(args, model, optimizer, criterion, dataloader, epoch):
    lossmeter_p = torch.tensor(0, device=args.device, dtype=torch.float32)
    lossmeter_t = torch.tensor(0, device=args.device, dtype=torch.float32)

    epoch_lr = adj_lr(args.lr, epoch, args.lr_decay_epoch, args.lr_decay_rate)
    optimizer.param_groups[0]['lr'] = epoch_lr

    model.train()

    for step, (samples, target, img_names) in enumerate(dataloader):
        optimizer.zero_grad()

        samples = samples.to(torch.float32).to(args.device)
        target = target.to(torch.float32).to(args.device)

        pred = model(samples)

        loss_pixel = criterion['pixel'](pred, target)
        pred_prob = torch.sigmoid(pred)

        if args.tloss_w > 0:
            loss_topo = criterion['topo'](pred_prob.cpu(), target.cpu(), img_names).to(args.device)
        else:
            loss_topo = torch.tensor(0.0, device=args.device)

        loss = loss_pixel + args.tloss_w * loss_topo

        loss.backward()
        optimizer.step()

        lossmeter_p += loss_pixel.detach()
        lossmeter_t += loss_topo.detach()

        if (step + 1) % args.print_freq == 0:
            print('[{}/{}]: Pixel Loss:{}/{}, Topo Loss:{}/{}'.format(
                (step + 1), len(dataloader),
                loss_pixel, lossmeter_p / (step + 1),
                args.tloss_w * loss_topo, args.tloss_w * lossmeter_t / (step + 1)
            ))

    lossmeter_p = (lossmeter_p / len(dataloader)).cpu()
    lossmeter_t = (lossmeter_t / len(dataloader)).cpu()

    return lossmeter_p, lossmeter_t


def validate(args, model, dataloader):
    accmeter = []
    dicemeter = []

    model.eval()
    with torch.no_grad():
        for step, (samples, target, img_names) in enumerate(dataloader):
            samples = samples.to(torch.float32).to(args.device)
            target = target.to(torch.float32).to(args.device)

            pred = model(samples)

            pred = (pred > 0.0).to(torch.float32)

            acc = pixel_accuracy(pred, target)
            dice = dice_score(pred, target)

            accmeter.extend(acc)
            dicemeter.extend(dice)

        acc_mean = np.mean(accmeter)
        dice_mean = np.mean(dicemeter)

    print('Acc:{}, Dice:{}'.format(acc_mean, dice_mean))
    return acc_mean, dice_mean


def trainer(args, model):
    wandb.init(
        entity="justin-shi9871-vanderbilt-university",
        project="topologial-loss-unet-small-train",
        name=args.exp,
        config={
            "learning_rate": args.lr,
            "architecture": args.model,
            "dataset": args.dataset,
            "epochs": args.epoch,
            "batch_size": args.batch_size,
            "tau": args.tau, 
            "alpha": args.alpha,
            "tloss_w": args.tloss_w,
            "precal_PD": args.precal_PD,
        }
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        betas=args.betas,
        eps=args.eps
    )

    criterion = {}
    pos_weight = torch.tensor([2298.0]).to(args.device)
    criterion['pixel'] = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion['topo'] = PDMatchingLoss(args)
    train_dataset = Topo_dataloader(os.path.join(args.dataroot, args.dataset), 'train')
    val_dataset = Topo_dataloader(os.path.join(args.dataroot, args.dataset), 'val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    if args.precal_PD:
        print('Pre-calculating the PD for ground truth to avoid repetition in training ...')
        for step, (samples, target, img_names) in enumerate(train_loader):
            if step % 10 == 0:
                print('{}/{}'.format(step, len(train_loader)))
            target = target.to(torch.float32)
            criterion['topo']._pre_compute_PD(target, img_names)

    best_val_metric = {'acc': 0,'dice': 0}
    trainlossmeter = {'pixel': [], 'topo': []}
    valmetricmeter = {'acc': [], 'dice': []}

    for epoch in range(args.start_epoch, args.epoch):
        print('===================== Epoch {} ====================='.format(epoch + 1))
        print('Training ...')

        train_loss = train_one_epoch(args, model, optimizer, criterion, train_loader, epoch)
        trainlossmeter['pixel'].append(train_loss[0].item())
        trainlossmeter['topo'].append(train_loss[1].item())

        print('Validating ...')
        val_metric = validate(args, model, val_loader)
        valmetricmeter['acc'].append(val_metric[0])
        valmetricmeter['dice'].append(val_metric[1])

        wandb.log({
            'epoch': epoch + 1,
            'train/pixel_loss': train_loss[0].item(),
            'train/topo_loss': train_loss[1].item(),
            'val/acc': val_metric[0],
            'val/dice': val_metric[1],
            'lr': optimizer.param_groups[0]['lr'],
        })

        if (epoch + 1) % args.save_model_interval == 0:
            print('Saving model @ epoch ', epoch + 1)
            torch.save(model.state_dict(), os.path.join(args.exp_output_dir, 'epoch{}.pth'.format(epoch + 1)))

        if val_metric[0] > best_val_metric['acc']:
            print('New best val accuracy: {} > {}, saving new best model'.format(
                val_metric[0], best_val_metric['acc']
            ))
            best_val_metric['acc'] = val_metric[0]
            torch.save(model.state_dict(), os.path.join(args.exp_output_dir, 'best_acc.pth'))

        if val_metric[1] > best_val_metric['dice']:
            print('New best val dice score: {} > {}, saving new best model'.format(
                val_metric[1], best_val_metric['dice']
            ))
            best_val_metric['dice'] = val_metric[1]
            torch.save(model.state_dict(), os.path.join(args.exp_output_dir, 'best_dice.pth'))

    print(trainlossmeter)
    print(valmetricmeter)
    print(best_val_metric)

    wandb.finish()