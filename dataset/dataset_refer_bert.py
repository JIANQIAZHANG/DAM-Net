import os
import sys
import torch.utils.data as data

from bert.tokenization_bert import BertTokenizer
from copy import deepcopy

import h5py
from refer.refer import REFER

from args import get_parser

import json
from . import transform_add

from .transform_add import *


# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()


def build_additional_strong_transform(args):
    # assert cfg.get("strong_aug", False) != False
    strong_aug_nums = args.num_augs
    strong_img_aug = transform_add.strong_img_aug(strong_aug_nums)
    return strong_img_aug


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False,
                 label=True,
                 nsample=None):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)
        self.label = label  # fcl add

        self.max_tokens = 20
        if self.split == "train":
            if self.label == False:
                self.stat_refs_list = json.loads(open(args.unlabeled_data, 'r').read().rstrip('\x00'))
                # self.stat_refs_list1 = json.loads(open(args.labeled_data, 'r').read().rstrip('\x00'))

                tmp = [item['mask_id'] for item in self.stat_refs_list['train']]
                # aaa = len(self.stat_refs_list1['train']) * 10

                # ref_ids = tmp[:aaa]
                ref_ids = tmp
            if self.label == True and nsample is not None:
                self.stat_refs_list = json.loads(open(args.labeled_data, 'r').read().rstrip('\x00'))
                tmp = [item['mask_id'] for item in self.stat_refs_list['train']]
                ref_ids = tmp
                ref_ids *= math.ceil(nsample / len(ref_ids))
                random.shuffle(ref_ids)
                ref_ids = ref_ids[:nsample]
        else:
            ref_ids = self.refer.getRefIds(split=self.split)

        # ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode

        self.StrongAug = build_additional_strong_transform(args)
        self.ToTensor = ToTensor()
        self.Normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]
        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])

        choice_sent = np.random.choice(len(ref[0]['sentences']))
        text = ref[0]['sentences'][choice_sent]['raw']

        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")
        # self.eval_mode 想进行更严谨的测试
        percent = random.random()
        if self.split == 'val':
            if self.image_transforms is not None:
                img, target, text, percent = self.image_transforms(img, annot, text, percent)

                attention_mask = [0] * self.max_tokens
                tensor_embeddings = [0] * self.max_tokens
                text_encoder = self.tokenizer.encode(text, add_special_tokens=True)
                text_encoder = text_encoder[:self.max_tokens]
                tensor_embeddings[:len(text_encoder)] = text_encoder
                attention_mask[:len(text_encoder)] = [1] * len(text_encoder)

                tensor_embeddings = torch.tensor(tensor_embeddings).unsqueeze(0)
                attention_mask = torch.tensor(attention_mask).unsqueeze(0)

                img, target = self.ToTensor(img, target)
                img, target = self.Normalize(img, target)
                return img, target, tensor_embeddings, attention_mask

        if self.split == 'train':
            if self.image_transforms is not None:
                # resize, from PIL to tensor, and mean and std normalization
                img_w, target, text_w, percent = self.image_transforms(img, annot, text, percent)
                attention_mask_w = [0] * self.max_tokens
                tensor_embeddings_w = [0] * self.max_tokens
                text_encoder_w = self.tokenizer.encode(text_w, add_special_tokens=True)
                text_encoder_w = text_encoder_w[:self.max_tokens]
                tensor_embeddings_w[:len(text_encoder_w)] = text_encoder_w
                attention_mask_w[:len(text_encoder_w)] = [1] * len(text_encoder_w)

                tensor_embeddings_w = torch.tensor(tensor_embeddings_w).unsqueeze(0)
                attention_mask_w = torch.tensor(attention_mask_w).unsqueeze(0)
            if self.label == True:
                img_l, target = self.ToTensor(img_w, target)
                img_l, target = self.Normalize(img_l, target)
                return img_l, target, tensor_embeddings_w, attention_mask_w
            else:
                if args.textAug:
                    aug_sentences = self.eda_aug(text_w)
                    text_s = aug_sentences

                    embeddings = []
                    attention_masks = []
                    for ttt in text_s:
                        attention_mask_q = [0] * self.max_tokens
                        tensor_embeddings_q = [0] * self.max_tokens
                        text_encoder_q = self.tokenizer.encode(ttt, add_special_tokens=True)
                        text_encoder_q = text_encoder_q[:self.max_tokens]
                        tensor_embeddings_q[:len(text_encoder_q)] = text_encoder_q
                        attention_mask_q[:len(text_encoder_q)] = [1] * len(text_encoder_q)
                        tensor_embeddings_q = torch.tensor(tensor_embeddings_q).unsqueeze(0)
                        attention_mask_q = torch.tensor(attention_mask_q).unsqueeze(0)
                        embeddings.append(tensor_embeddings_q.unsqueeze(0))
                        attention_masks.append(attention_mask_q.unsqueeze(0))

                    tensor_embeddings_s = torch.cat(embeddings, dim=1)
                    attention_mask_s = torch.cat(attention_masks, dim=1)
                    tensor_embeddings_s = tensor_embeddings_s.permute(0, 2, 1)
                    attention_mask_s = attention_mask_s.permute(0, 2, 1)
                else:
                    tensor_embeddings_s1 = deepcopy(tensor_embeddings_w)
                    tensor_embeddings_s2 = deepcopy(tensor_embeddings_w)
                    attention_mask_s1 = deepcopy(attention_mask_w)
                    attention_mask_s2 = deepcopy(attention_mask_w)

                modifer = 0.999
                ignore_mask = Image.fromarray(np.zeros((target.size[1], target.size[0])))
                # array.shape 第一个元素是高，第二个元素是宽
                # PIL.size  第一个元素是宽，第二个元素是高
                # img_u = self.ColorJitter(img)
                img_w_tensor, _ = self.ToTensor(img_w, target)
                img_w_tensor, _ = self.Normalize(img_w_tensor, target)

                img_s_1 = deepcopy(img_w)
                img_s_2 = deepcopy(img_w)

                img_s_1 = self.StrongAug(img_s_1, modifer)
                img_s_2 = self.StrongAug(img_s_2, modifer)

                img_s_1_tensor, ignore_mask = self.ToTensor(img_s_1, ignore_mask)
                img_s_1_tensor, ignore_mask = self.Normalize(img_s_1_tensor, ignore_mask)

                img_s_2_tensor, _ = self.ToTensor(img_s_2, ignore_mask)
                img_s_2_tensor, _ = self.Normalize(img_s_2_tensor, ignore_mask)

                mask = torch.from_numpy(np.array(target)).long()
                ignore_mask[mask == 255] = 255
                return img_w_tensor, img_s_1_tensor, img_s_2_tensor, ignore_mask, tensor_embeddings_w, attention_mask_w, tensor_embeddings_s1, attention_mask_s1, tensor_embeddings_s2, attention_mask_s2

