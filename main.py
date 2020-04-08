'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from models import ResNet18
from utils import AverageMeter, accuracy

from utils.tricks import mixup_data, mixup_criterion
from utils.tricks import CrossEntropyLossMaybeSmooth
from utils.tricks import GradualWarmupScheduler


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Experiment setting
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60],
                        help='decrease learning rate at these epochs')
    parser.add_argument('--epochs', type=int, default=90,
                        help="total epochs for training")
    parser.add_argument('--workers', type=int, default=4,
                        help="number of worker to load data")
    parser.add_argument('--gpu', action='store_true',
                        help="whether to use gpu")
    parser.add_argument('--save_model', type=str,
                        default='best.pt', help="path to save model")

    # Bag of tricks
    parser.add_argument('--smooth', type=float, default=0.0,
                        help="eps for label smoothing, 0 to disable")
    parser.add_argument('--warmup', type=int, default=0,
                        help="warmup epoch, 0 to disable")
    parser.add_argument('--warmup_lr', type=float,
                        default=1e-5, help="initial learning rate")
    parser.add_argument('--cosine_lr', action='store_true',
                        help="whether to use cosine learning rate")
    parser.add_argument('--mixup', type=float, default=0.0,
                        help="alpha for mixup training, 0 to disable")

    args = parser.parse_args()

    assert 0.0 <= args.smooth <= 1
    assert args.warmup >= 0

    args.do_mixup = args.mixup > 0.0
    args.do_smooth = args.smooth > 0.0
    args.do_warmup = args.warmup > 0

    return args


def train(dataloader, criterion, optimizer, epoch, model, scheduler, args):
    # switch to train mode
    model.train()

    if args.do_smooth:
        criterion.train()

    losses = AverageMeter()
    accs = AverageMeter()

    for i, (data, target) in enumerate(dataloader):

        if args.gpu:
            data = data.cuda()
            target = target.cuda()

        if args.do_mixup:
            data, target_a, target_b, lam = mixup_data(
                data, target, args.mixup)

        output = model(data)

        if args.do_mixup:
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            target = target_a if lam >= 0.5 else target_b  # For more precise accuracy shown
        else:
            loss = criterion(output, target)

        acc, *_ = accuracy(output, target, topk=(1,))

        losses.update(loss.item(), data.size(0))
        accs.update(acc[0].item(), data.size(0))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        scheduler.step()

        print(f'\rEpoch: [{epoch}][{i}/{len(dataloader)}]'
              f'  Loss {losses.val:.4f} ({losses.avg:.4f})'
              f'  Acc@1 {accs.val:.3f} ({accs.avg:.3f})'
              f'  LR {optimizer.param_groups[0]["lr"]:.5f}', end=' ')
    print()


def validate(dataloader, criterion, model, epoch, best_acc, args):
    # switch to evaluate mode
    model.eval()

    if args.do_smooth:
        criterion.eval()

    accs = AverageMeter()
    losses = AverageMeter()

    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            if args.gpu is not None:  # TODO None?
                data = data.cuda()
                target = target.cuda()

            # compute output
            output = model(data)

            # measure accuracy and record loss
            acc, *_ = accuracy(output, target, topk=(1,))
            accs.update(acc[0], data.size(0))

            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))

            print(f'\rEpoch: [{epoch}][{i}/{len(dataloader)}]'
                  f'  Loss {losses.val:.4f} ({losses.avg:.4f})'
                  f'  Acc@1 {accs.val:.3f} ({accs.avg:.3f})', end=' ')

    print(f'\nAcc@1 {accs.avg:.3f}')

    if accs.avg.item() > best_acc:
        print(f'new best_acc is {accs.avg:.3f}')
        print(f'saving model {args.save_model}')
        torch.save(model.state_dict(), args.save_model)
        best_acc = accs.avg.item()
    print()

    return best_acc


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    args = parse_args()
    set_seed()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.workers)

    model = ResNet18()

    if args.do_smooth:
        criterion = CrossEntropyLossMaybeSmooth(args.smooth)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    init_lr = args.warmup_lr if args.do_warmup else args.lr

    optimizer = torch.optim.SGD(model.parameters(), init_lr)

    if args.cosine_lr:
        epochs = args.epochs - args.warmup
        scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs*len(trainloader), eta_min=4e-8)
    else:
        milestones = [(e-args.warmup)*len(trainloader)
                      for e in args.schedule]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    if args.do_warmup:
        total_iter = args.warmup*len(trainloader)
        multiplier = args.lr/args.warmup_lr
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=multiplier, total_iter=total_iter, after_scheduler=scheduler)

    best_acc = 0
    for epoch in range(args.epochs):
        train(trainloader, criterion, optimizer, epoch, model, scheduler, args)
        best_acc = validate(testloader, criterion,
                            model, epoch, best_acc, args)
        print(f'Best acc is {best_acc:.3f}')


if __name__ == '__main__':
    main()
