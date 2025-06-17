import logging
import os

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from util.utils import count_params, init_log_save, AverageMeter
from util.dist_helper import setup_distributed

import random
from model.lavt import segmentation
from functools import reduce
import operator
import dataset.transform_add as T

from valid_save import evaluate_save

from args import get_parser  # fcl add
parser = get_parser()  # fcl add
args = parser.parse_args()

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


def main():
    args = parser.parse_args()

    # cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)   #----------------

    logger = init_log_save(args.save_path, 'global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True               # cudnn 深度学习优化算法
    # cudnn.enabled = True

    # model initialization
    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    if args.local_rank <= 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False, find_unused_parameters=True)
    single_model = model.module

    # ------- #
    # dataset #
    # ------- #
    trainset_u, num_classes = get_dataset("train", get_transform(args=args), args=args, label=False)

    trainset_l, _ = get_dataset("train", get_transform(args=args), args=args, label=True, nsample=len(trainset_u))

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
    """
        保持相同的学习率去优化，才更有可比性。
    """
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
    # # # learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(trainloader_u) * args.epochs)) ** 0.9)

    # loss
    # CE loss for labeled data
    criterion_l = nn.CrossEntropyLoss(reduction='mean', ignore_index=255, weight=torch.FloatTensor([0.5, 1.5]).cuda(args.local_rank)).cuda(args.local_rank)

    total_iters = len(trainloader_u) * args.epochs
    previous_best = 0.0
    
    from tqdm import tqdm
    for epoch in tqdm(range(args.epochs)):
        if rank == 0:
            logger.info('===========> Epoch: {:}, backbone LR: {:.8f}, segmentation LR: {:.8f}, Previous best: {:.2f}'.format(epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[2]['lr'],previous_best))

        model.train()
        # total_loss = AverageMeter()
        total_loss = 0.0
        trainloader_l.sampler.set_epoch(epoch)

        for i, (labeled_img, labeled_img_mask, labeled_sentences, labeled_attentions) in tqdm(enumerate(trainloader_l)):

            labeled_img, labeled_img_mask, labeled_sentences, labeled_attentions = labeled_img.cuda(
                args.local_rank), labeled_img_mask.cuda(args.local_rank), labeled_sentences.cuda(
                args.local_rank), labeled_attentions.cuda(args.local_rank)

            labeled_sentences = labeled_sentences.squeeze(1)
            labeled_attentions = labeled_attentions.squeeze(1)

            output = model(labeled_img, labeled_sentences, l_mask=labeled_attentions)

            # 如果考虑进行增强的话，这里也得计算需要忽略掉的像素
            loss = criterion_l(output, labeled_img_mask)

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # total_loss.update(loss.item())
            total_loss += loss.item()

            iters = epoch * len(trainloader_l) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr * args.lr_backbone
            optimizer.param_groups[1]["lr"] = lr * args.lr_backbone
            optimizer.param_groups[2]["lr"] = lr * args.lr_network
            optimizer.param_groups[3]["lr"] = lr * args.lr_backbone

            # if (i % args.val_freq == 0) and (rank == 0):
            #     logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss / (i + 1)))
            # if (i % args.val_freq == 0) and (rank == 0):
            #     evaluate_result = evaluate_save(args.local_rank, model, valloader, num_classes)
            #
            #     oIOU = evaluate_result['IOU']
            #     writer.add_scalar('oIOU_branch1', oIOU, epoch)
            #
            #     logger.info('***** Evaluation the model ***** >>>> oIOU: {:.2f}\n'.format(oIOU))
            #
            #
            #     if oIOU > previous_best:
            #         if previous_best != 0:
            #             os.remove(os.path.join(args.save_path, '%s_%.2f.pth' % (args.backbone, previous_best)))
            #         previous_best = oIOU
            #         torch.save(model.module.state_dict(),
            #                    os.path.join(args.save_path, '%s_%.2f.pth' % (args.backbone, oIOU)))


            if rank == 0:
                writer.add_scalar('train_loss_total', total_loss / (i+1), iters)
                # writer.add_scalar('train/loss_x', loss.item(), iters)

            if (i % (max(2, len(trainloader_l) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss / (i + 1)))

        if rank == 0:
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
    main()
