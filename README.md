# DAM-Net: Dual-Stream and Attention-Guided Multimodal Network for Semi-Supervised Referring Segmentation
This repository contains the implementation and resources for the paper titled "DAM-Net: Dual-Stream and Attention-Guided Multimodal Network for Semi-Supervised Referring Segmentation" by Huanhuan Zhou, Jianqi Zhang, Ling Bai, Tingmin Liu, and Ying Zang.

## Overview

DAM-Net is a novel deep learning framework designed for semi-supervised referring segmentation tasks. It utilizes a dual-stream architecture and attention-guided mechanisms to effectively segment objects referred to in natural language descriptions.
## Getting Started

### Installation

```
conda create -n DAM-Net python=3.8.4
conda activate DAM-Net
pip install -r requirements.txt
```

### Dataset

COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip) | [val2017](http://images.cocodataset.org/zips/val2017.zip) | [masks](https://drive.google.com/file/d/166xLerzEEIbU7Mt1UGut-3-VN41FMUb1/view?usp=sharing)

## Usage

```
python unimatch.py --backbone swin_tiny --save_path  /exp  --nodes 1 --port 12345 --gpus 4  --epochs 40 --batch_size 4  --w_CE 5.0 --w_con 2.0 --num_workers 8  --dataset refcoco --splitBy unc  --labeled_data path/refcoco_10%_image.json --unlabeled_data path/refcoco_90%_image.json

```

