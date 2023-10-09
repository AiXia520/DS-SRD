from __future__ import print_function, division

import sys
import time
import torch
import torch.nn as nn
from .util import AverageMeter, accuracy
import numpy as np
from helper.util import warmup_learning_rate
import torch.nn.functional as F
from random import sample

from crd.dkd_loss import dkd_loss

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def mixup(input, alpha, share_lam=False):
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha, alpha]
    beta = torch.distributions.beta.Beta(*alpha)
    randind = torch.randperm(input.shape[0], device=input.device)
    if share_lam:
        lam = beta.sample().to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam
    else:
        lam = beta.sample([input.shape[0]]).to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam.view([-1] + [1]*(input.dim()-1))
    output = lam_expanded * input + (1. - lam_expanded) * input[randind]
    return output, randind, lam


@torch.no_grad()
def sinkhorn(out):
    Q = torch.exp(out / 0.05).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1]   # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(3):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


def train_distill_ds(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2].cuda()

    model_s = module_list[0].cuda()  #weak student
    model_st = module_list[1].cuda()  # strong student
    model_t = module_list[2].cuda()  # teacher

    model_t.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)
        if epoch < opt.warm_epochs:
            warmup_learning_rate(optimizer, epoch, idx, len(train_loader), opt)
        im_q = input[0].float().cuda()
        im_k = input[1].float().cuda()
        im_q_origin = im_q
        target = target.cuda()
        index = index.cuda()
        if opt.distill in ['crd']:
            contrast_idx = contrast_idx.cuda()

        #mixup
        # bsz = im_q.shape[0]
        # im_q, labels_aux, lam = mixup(im_q, 1.0)

        preact = False
        # feat_st, logit_st = model_st(im_q, is_feat=True) #strong student

        feat_s, logit_s = model_s(im_q_origin, is_feat=True, preact=preact)  #weak student

        with torch.no_grad():
            feat_t, logit_t = model_t(im_k, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]


        # logits_mix = nn.functional.normalize(logit_st, dim=1).mm(nn.functional.normalize(logit_t, dim=1).t())
        # logits_mix /= 0.2
        # target_logits_mix = lam * logits_mix.diag() + (1. - lam) * logits_mix[range(bsz), labels_aux]
        # loss_mix =(2. - 2. * target_logits_mix).mean()

        #mixup

        l = np.random.beta(1.0, 1.0)
        idxa = torch.randperm(im_q.size(0))
        image_a, image_b = im_q, im_q[idxa]
        label_a, label_b = target, target[idxa]
        mixed_image = l * image_a + (1 - l) * image_b
        feat_st, logit_st = model_st(mixed_image, is_feat=True)  # strong student
        loss_mix = l * criterion_cls(logit_st, label_a) + (1 - l) * criterion_cls(logit_st, label_b)

        # cls + loss_div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # crd
        # f_s = feat_s[-1]
        # f_t = feat_t[-1]
        # loss_kd = criterion_kd(nn.functional.normalize(logit_s, dim=1), nn.functional.normalize(logit_t, dim=1), index, contrast_idx)

        # loss = opt.gamma * loss_cls + opt.alpha * loss_mix + opt.beta * loss_kd +loss_div
        loss = opt.gamma * loss_cls + opt.alpha * loss_mix + loss_div

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input[0].size(0))
        top1.update(acc1[0], input[0].size(0))
        top5.update(acc5[0], input[0].size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for param_q, param_k in zip(model_st.base_encoder.parameters(), model_t.parameters()):
                param_k.data = param_k.data * 0.99 + param_q.data * (1. - 0.99)

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill_pcl(epoch, train_loader, module_list, criterion_list, optimizer, opt, cluster_result = None):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        target = target.cuda()
        index = index.cuda()
        if opt.distill in ['crd']:
            contrast_idx = contrast_idx.cuda()

        im_q = input[0].float().cuda()
        im_k = input[1].float().cuda()

        #mixup
        # l = np.random.beta(1.0, 1.0)
        # idxa = torch.randperm(im_k.size(0))
        # image_a, image_b = im_k, im_k[idxa]
        # label_a, label_b = target, target[idxa]
        # mixed_image = l * image_a + (1 - l) * image_b

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(im_q, is_feat=True, preact=preact)
        feat_ss, logit_ss = model_s(im_k, is_feat=True , preact=preact)

        if opt.model_s in ['wrn_40_1', 'ShuffleV1', 'ShuffleV2','MobileNetV2', 'vgg8']:
            q_norm = nn.functional.normalize(logit_s, dim=1)
            k_norm = nn.functional.normalize(logit_ss, dim=1)
        else:
            q_norm = nn.functional.normalize(feat_s[-1], dim=1)
            k_norm = nn.functional.normalize(feat_ss[-1], dim=1)

        with torch.no_grad():
            feat_t, logit_t = model_t(im_k, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)

        if opt.dkd:
            loss_div = min(epoch / 20, 1.0) * dkd_loss(logit_s, logit_t, target, opt.dkd_alpha, opt.dkd_beta, opt.dkd_t)
        else:
            loss_div = criterion_div(logit_s, logit_t)
        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        # local structured distillation
        loss_proto = 0
        if cluster_result is not None:
            proto_student = []
            proto_teacher = []
            for n, (im2cluster, prototypes, density) in enumerate(
                    zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])):
                # get positive prototypes
                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]

                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max())]
                neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
                neg_proto_id = sample(neg_proto_id, 100)  # sample r negative prototypes
                neg_prototypes = prototypes[neg_proto_id]

                proto_selected = torch.cat([pos_prototypes, neg_prototypes], dim=0)

                # compute prototypical logits

                student = torch.mm(q_norm, proto_selected.t())
                teacher = torch.mm(k_norm, proto_selected.t())

                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id, torch.LongTensor(neg_proto_id).cuda()], dim=0)]

                student /= temp_proto
                teacher /= temp_proto
                # loss_consist += F.kl_div(student, teacher, reduction='mean')
                proto_student.append(student)
                proto_teacher.append(teacher)
            # ProtoNCE loss
            for proto_out, proto_target in zip(proto_student, proto_teacher):
                loss_proto += criterion_div(proto_out, proto_target)
            # average loss across all sets of prototypes
            loss_proto /= len(opt.num_cluster)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd + opt.betac * loss_proto
        # loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_proto
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input[0].size(0))
        top1.update(acc1[0], input[0].size(0))
        top5.update(acc5[0], input[0].size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg





def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
