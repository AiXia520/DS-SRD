from __future__ import print_function

import torch
import numpy as np
import math

def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    # steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    # if steps > 0:
    #     new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)

    lr = opt.learning_rate
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / opt.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, epoch, batch_id, total_batches, opt):
    p = (batch_id + 1 + epoch * total_batches) / (opt.warm_epochs * total_batches)
    lr = opt.warmup_from + p * (opt.warmup_to - opt.warmup_from)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    pass
