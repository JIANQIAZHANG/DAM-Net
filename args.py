import argparse

def str2bool(v):                    # 更容易的处理布尔选项，增加对用户友好性和输入数据的灵活性
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser(description='Weak-to-Strong Consistency in RES')

    parser.add_argument('--save_path', default="/exp/", type=str)

    parser.add_argument('--model', default='lavt_one', help='model: lavt, lavt_one')
    parser.add_argument('--pretrained_swin_weights', default='path/datasets/pretrained_weights/swin_tiny_patch4_window7_224_22k.pth', help='path to pre-trained Swin backbone weights')
    parser.add_argument('--seed', default=22, type=int)
    # lavt  fcl add
    parser.add_argument('--swin_type', default='tiny', help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--window12', action='store_true', default=False,
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--bert_tokenizer', default='/path/datasets/bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--ck_bert', default='/path/datasets/bert-base-uncased', help='pre-trained BERT weights')
    parser.add_argument('--refer_data_root', default='/path/datasets', help='REFER dataset root directory')
    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')

    # ddp settings
    parser.add_argument('--gpus', default=4, type=int, help='number of gpus per node')
    parser.add_argument('--nodes', default=1, type=int, help='number of nodes')
    parser.add_argument("--ddp", default=True, type=str2bool, help='distributed data parallel training or not')
    # default setting
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument('--val_freq', default=100, type=int, help='print freq')
    parser.add_argument('--port', default=2020, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr', default=0.00005, type=float, help='the initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay', dest='weight_decay')
    parser.add_argument('--amsgrad', action='store_true', help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('--lr_network', default=1.0, type=float)                            # coefficient of the lr of other modules of the model
    parser.add_argument('--lr_backbone', default=1.0, type=float)                           # coefficient of the lr of the backbone of the model
    parser.add_argument('--mul_scheduler', default=0.9, type=float)                         # coefficient of the exp scheduler
    parser.add_argument('--backbone', default='swin_tiny', type=str)
    parser.add_argument('--num_augs', default=3, type=int, help='for strong Aug')
    parser.add_argument('--w_CE', default=5.0, type=float)
    parser.add_argument('--w_con', default=2.0, type=float)

    # EDA_Augment
    parser.add_argument("--num_aug", default=9, type=int, help="number of augmented sentences per original sentence")
    parser.add_argument("--alpha_sr", default=0.1, type=float,help="percent of words in each sentence to be replaced by synonyms")
    parser.add_argument("--alpha_ri", default=0.1, type=float, help="percent of words in each sentence to be inserted")
    parser.add_argument("--alpha_rs", default=0.1, type=float, help="percent of words in each sentence to be swapped")
    parser.add_argument("--alpha_rd", default=0.1, type=float, help="percent of words in each sentence to be deleted")

    # quality_con
    parser.add_argument('--unsupervised_threshold', default=0.7, type=float)


    
    # data
    parser.add_argument('--labeled_data', default='path/anns/refcoco/refcoco_0-1%_image.json', help='path to labeled data')
    parser.add_argument('--unlabeled_data', default='path/anns/refcoco/refcoco_90%_image.json', help='path to unlabeled data')

    return parser