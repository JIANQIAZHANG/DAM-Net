import logging
import os

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from util.utils import count_params, init_log_save, AverageMeter
from util.dist_helper import setup_distributed

from args import get_parser  # fcl add
import random
from model.lavt import segmentation
from functools import reduce
import operator
import dataset.transform_add as T
from valid_save import evaluate_save

import torch.multiprocessing as mp

from util.losses import DiceLoss
from util import ramps

import numpy as np

import random

parser = get_parser()  # fcl add
args = parser.parse_args()

args.world_size = args.gpus * args.nodes

def get_dataset(image_set, transform, args, label=True, nsample=None):
    from dataset.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      label=label,
                      nsample=nsample)
    num_classes = 2

    return ds, num_classes

def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  # T.ToTensor(),
                  # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)

def get_transform_aug(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.RandomHorizontalFlip(),
                  ]

    return T.Compose(transforms)

def get_current_consistency_weight(iters, total_iters):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * ramps.sigmoid_rampup(iters, total_iters)  

import copy
def ssmix_nosal(input1, input2, length1, length2, max_len, cur_quality):
    inputs_aug = copy.deepcopy(input1)
    ratio = torch.ones((len(length1),)).to(input1['sents'].device)

    for idx in range(len(length1)):
        ss_weight = cur_quality[idx]
        if length1[idx].item() > max_len:
            for key in inputs_aug.keys():
                inputs_aug[key][idx][max_len:] = 0
            inputs_aug['sents'][idx][max_len - 1] = 102  # artificially add EOS token.  #  # j
        l1, l2 = min(length1[idx].item(), max_len), length2[idx].item()

        # self.args.ss_winsize = 10
        # if self.args.ss_winsize == -1:
        #     window_size = random.randrange(0, l1)  # random sampling of window_size
        # else:
        #     # remove EOS & SOS when calculating ratio & window size.
        #     window_size = int((l1 - 2) * self.args.ss_winsize / 100.) or 1

        # opt direction
        window_size = int((l1 - 2) * (1 - ss_weight)) or 1

        if l2 <= window_size:
            ratio[idx] = 1
            continue

        start_idx = random.randrange(0, l1 - window_size)  # random sampling of starting point
        if (l2 - window_size) < start_idx:  # not enough text for reference.
            ratio[idx] = 1
            continue
        else:
            ref_start_idx = start_idx
        mix_percent = float(window_size) / (l1 - 2)

        for key in input1.keys():
            inputs_aug[key][idx, start_idx:start_idx + window_size] = \
                input2[key][idx, ref_start_idx:ref_start_idx + window_size]

        ratio[idx] = 1 - mix_percent
    return inputs_aug, ratio
  
  
def main(gpu, ngpus_per_node, args):
    # args = parser.parse_args()
    args.local_rank = gpu
    if args.local_rank <= 0:
        os.makedirs(args.save_path, exist_ok=True)
    # cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)   #----------------
    logger = init_log_save(args.save_path, 'global', logging.INFO)
    logger.propagate = 0

    # rank, world_size = setup_distributed(port=args.port)

    if args.local_rank <= 0:
        writer = SummaryWriter(args.save_path)
        # os.makedirs(args.save_path, exist_ok=True)

    if args.ddp:
        torch.distributed.init_process_group(backend='gloo', rank=args.local_rank, world_size=args.world_size)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True               
    # cudnn.enabled = True

    # model initialization
    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    if args.local_rank <= 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    # local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False, find_unused_parameters=True)
    single_model = model.module
    
    #if args.local_rank <= 0:
    # logger.info('load saved model')
    #checkpoint = torch.load(args.load_path, map_location='cpu')
    #model.module.load_state_dict(checkpoint)
    #print("load saved model")

    # ------- #
    # dataset #
    # ------- #
    trainset_u, num_classes = get_dataset("train", get_transform_aug(args=args), args=args, label=False)

    trainset_l, _ = get_dataset("train", get_transform_aug(args=args), args=args, label=True, nsample=len(trainset_u))

    valset, _ = get_dataset("val", get_transform(args=args), args=args, label=True)

    if args.ddp:
        trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    else:
        trainsampler_l = None
    trainloader_l = DataLoader(trainset_l, batch_size=args.batch_size,
                               pin_memory=True, num_workers=args.num_workers, drop_last=True, sampler=trainsampler_l)

    if args.ddp:
        trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    else:
        trainsampler_u = None

    trainloader_u = DataLoader(trainset_u, batch_size=args.batch_size,
                               pin_memory=True, num_workers=args.num_workers, drop_last=True, sampler=trainsampler_u)

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=args.num_workers, drop_last=False)

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0, 'lr': args.lr * args.lr_backbone},
        {'params': backbone_decay, 'lr': args.lr * args.lr_backbone},
        {"params": [p for p in single_model.classifier.parameters() if p.requires_grad], 'lr': args.lr * args.lr_network},
        # the following are the parameters of bert
        {"params": reduce(operator.concat, [[p for p in single_model.text_encoder.encoder.layer[i].parameters() if p.requires_grad] for i in range(10)]), 'lr': args.lr * args.lr_backbone},
    ]

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )
    # # learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(trainloader_u) * args.epochs)) ** 0.9)
    # print("")

    # loss
    l2_loss = nn.MSELoss()
    # CE loss for labeled data
    criterion_l = nn.CrossEntropyLoss(reduction='mean', ignore_index=255, weight=torch.FloatTensor([0.5, 1.5]).cuda(args.local_rank)).cuda(args.local_rank)

    # consistency loss for unlabeled data
    #dice_loss = DiceLoss(num_classes).cuda(args.local_rank)
    criterion_u = nn.CrossEntropyLoss(reduction='none', weight=torch.FloatTensor([0.5, 1.5]).cuda(args.local_rank)).cuda(args.local_rank)

    total_iters = len(trainloader_u) * args.epochs
    previous_best = 0.0
    


    from tqdm import tqdm
    for epoch in tqdm(range(args.epochs)):
        if args.local_rank <= 0:
            # logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(epoch, optimizer.param_groups[0]['lr'], previous_best))
            logger.info(
                '===========> Epoch: {:}, backbone LR: {:.8f}, segmentation LR: {:.8f}, Previous best: {:.2f}'.format(
                    epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[2]['lr'], previous_best))
        total_loss, total_loss_CE, total_loss_con = 0.0, 0.0, 0.0
        total_mask_ratio = 0.0

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        total_labeled = 0
        total_unlabeled = 0

        for i, ((labeled_img, labeled_img_mask, labeled_sentences, labeled_attentions),
                (unlabeled_img_w, unlabeled_img_s1, unlabeled_img_s2, ignore_img_mask, unlabeled_sentences_w,
                 unlabeled_attentions_w, unlabeled_sentences_s1, unlabeled_attentions_s1, unlabeled_sentences_s2,
                 unlabeled_attentions_s2)) in enumerate(loader):
            labeled_img, labeled_img_mask, labeled_sentences, labeled_attentions = labeled_img.cuda(
                args.local_rank), labeled_img_mask.cuda(args.local_rank), labeled_sentences.cuda(args.local_rank
                                                                                                 ), labeled_attentions.cuda(
                args.local_rank)
            unlabeled_img_w, unlabeled_img_s1, ignore_img_mask, unlabeled_sentences_w, unlabeled_attentions_w, unlabeled_sentences_s1, unlabeled_attentions_s1 = unlabeled_img_w.cuda(
                args.local_rank
                ), unlabeled_img_s1.cuda(args.local_rank), ignore_img_mask.cuda(
                args.local_rank), unlabeled_sentences_w.cuda(args.local_rank
                                                             ), unlabeled_attentions_w.cuda(
                args.local_rank), unlabeled_sentences_s1.cuda(args.local_rank
                                                              ), unlabeled_attentions_s1.cuda(args.local_rank)
            unlabeled_img_s2 = unlabeled_img_s2.cuda(args.local_rank)
            unlabeled_sentences_s2 = unlabeled_sentences_s2.cuda(args.local_rank)
            unlabeled_attentions_s2 = unlabeled_attentions_s2.cuda(args.local_rank)

            # if (i % args.val_freq == 0) and (args.local_rank <= 0):
            #     logger.info(
            #         '===========> Epoch: {:}, backbone LR: {:.8f}, segmentation LR: {:.8f}'.format(
            #             epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[2]['lr']))

            labeled_sentences = labeled_sentences.squeeze(1)
            labeled_attentions = labeled_attentions.squeeze(1)
            unlabeled_sentences_w = unlabeled_sentences_w.squeeze(1)
            unlabeled_attentions_w = unlabeled_attentions_w.squeeze(1)
            unlabeled_sentences_s1 = unlabeled_sentences_s1.squeeze(1)
            unlabeled_attentions_s1 = unlabeled_attentions_s1.squeeze(1)
            unlabeled_sentences_s2 = unlabeled_sentences_s2.squeeze(1)
            unlabeled_attentions_s2 = unlabeled_attentions_s2.squeeze(1)

            # generate pseudo labels and Mask-aware Confidence Score
            p_threshold = args.unsupervised_threshold

            with torch.no_grad():
                model.eval()

                pred_u_w,_ = model(unlabeled_img_w, unlabeled_sentences_w, l_mask=unlabeled_attentions_w)
                # conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]


            model.train()
            optimizer.zero_grad()
            torch.distributed.barrier()

            num_lb, num_ulb = labeled_img.shape[0], unlabeled_img_w.shape[0]
            total_labeled += num_lb
            total_unlabeled += num_ulb

            image_u_aug1 = unlabeled_img_s1
            image_u_aug2 = unlabeled_img_s2

            num_lb, num_ulb = labeled_img.shape[0], unlabeled_img_w.shape[0]
            total_labeled += num_lb
            total_unlabeled += num_ulb

            pred_l, attn_labeled = model(labeled_img, labeled_sentences, l_mask=labeled_attentions)

            # pred_u_ss1, pred_u_ss2 = model(torch.cat((image_u_aug1, image_u_aug2), dim=0),
            #                              torch.cat((unlabeled_sentences_s1, unlabeled_sentences_s2), dim=0),
            #                              l_mask=torch.cat((unlabeled_attentions_s1, unlabeled_attentions_s2),
            #                                               dim=0)).chunk(2)

            pred_u_ss, attn_unlabeleds = model(torch.cat((image_u_aug1, image_u_aug2), dim=0),
                                           torch.cat((unlabeled_sentences_s1, unlabeled_sentences_s2), dim=0),
                                           l_mask=torch.cat((unlabeled_attentions_s1, unlabeled_attentions_s2),
                                                            dim=0))

            pred_u_s1, pred_u_s2 = pred_u_ss.chunk(2)
            attn_unlabeled1, attn_unlabeled2 = attn_unlabeleds.chunk(2)
            # pred_u_s1, attn_unlabeled1 = pred_u_ss1[0], pred_u_ss1[1]
            # pred_u_s2, attn_unlabeled2 = pred_u_ss2[0], pred_u_ss2[1]


            # pred_u_s, attn_unlabeled = model(image_u_aug, unlabeled_sentences_q, l_mask=unlabeled_attentions_q)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            loss_attn1 = l2_loss(attn_unlabeled1, attn_labeled)
            loss_attn2 = l2_loss(attn_unlabeled2, attn_labeled)

            loss_attn = (loss_attn1 + loss_attn2)/2
            loss_CE = criterion_l(pred_l, labeled_img_mask)
			
            # consistency loss
            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w)
            loss_u_s1 = loss_u_s1 * ((conf_u_w >= p_threshold) & (ignore_img_mask != 255))

            loss_u_s1 = loss_u_s1.sum(dim=[1,2]) / (labeled_img_mask != 255).sum(dim=[1,2])
            loss_u_s1 = loss_u_s1.mean()

            # consistency loss
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w)
            loss_u_s2 = loss_u_s2 * ((conf_u_w >= p_threshold) & (ignore_img_mask != 255))

            loss_u_s2 = loss_u_s2.sum(dim=[1, 2]) / (labeled_img_mask != 255).sum(dim=[1, 2])
            loss_u_s2 = loss_u_s2.mean()


            loss_u_s = (loss_u_s1 + loss_u_s2)/2

            loss = loss_CE * args.w_CE + loss_u_s * args.w_con + loss_attn * args.w_con


            torch.distributed.barrier()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_CE += loss_CE.item()
            total_loss_con += loss_u_s.item()

            iters = epoch * len(trainloader_u) + i

            mask_ratio = ((conf_u_w >= 0.95) & (ignore_img_mask != 255)).sum().item() / \
                (ignore_img_mask != 255).sum()
            total_mask_ratio += mask_ratio.item()

            # update lr
            # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda x: (1 - x / (len(trainloader_u) * args.epochs)) ** 0.9)
            backbone_lr = args.lr * (1 - iters / total_iters) ** args.mul_scheduler  
            backbone_lr = backbone_lr * args.lr_backbone

            seg_lr = args.lr * (1 - iters / total_iters) ** args.mul_scheduler
            seg_lr = seg_lr * args.lr_network

            optimizer.param_groups[0]["lr"] = backbone_lr
            optimizer.param_groups[1]["lr"] = backbone_lr
            optimizer.param_groups[2]["lr"] = seg_lr
            optimizer.param_groups[3]["lr"] = backbone_lr

            if (i % (len(trainloader_u) // 8) == 0) and args.local_rank <= 0:
                writer.add_scalar('train/loss_all', total_loss / (i + 1), iters)
                writer.add_scalar('train/loss_x', total_loss_CE / (i + 1), iters)
                writer.add_scalar('train/loss_s', total_loss_con / (i + 1), iters)
                writer.add_scalar('train/mask_ratio', total_mask_ratio/ (i + 1), iters)

            if (i % (len(trainloader_u) // 8) == 0) and (args.local_rank <= 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss / (i + 1), total_loss_CE / (i + 1), total_loss_con / (i + 1), total_mask_ratio / (i + 1)))

        if args.local_rank <= 0:
            evaluate_result = evaluate_save(args.local_rank, model, valloader, num_classes)

            oIOU = evaluate_result['IOU']
            writer.add_scalar('oIOU_branch1', oIOU, epoch)

            logger.info('***** Evaluation the model ***** >>>> oIOU: {:.2f}\n'.format(oIOU))

            if oIOU > previous_best:
                if previous_best != 0:
                    os.remove(os.path.join(args.save_path, '%s_%.2f.pth' % (args.backbone, previous_best)))
                previous_best = oIOU
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_path, '%s_%.2f.pth' % (args.backbone, oIOU)))

if __name__ == '__main__':

    if args.ddp:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(args.port)
        # args.base_lr = args.base_lr * args.world_size

    if args.ddp:
        mp.spawn(main, nprocs=args.gpus, args=(args.gpus, args))
    else:
        main(-1, 1, args)
