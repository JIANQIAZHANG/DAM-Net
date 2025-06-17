import random
import math

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
#
from torchvision.transforms import functional as F
import torchvision.transforms as T


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.rand_brightness = RandomBrightness(brightness)
        self.rand_contrast = RandomContrast(contrast)
        self.rand_saturation = RandomSaturation(saturation)

    def __call__(self, img):
        if random.random() < 0.8:
            image = img
            func_inds = list(np.random.permutation(3))
            for func_id in func_inds:
                if func_id == 0:
                    image = self.rand_brightness(image)
                elif func_id == 1:
                    image = self.rand_contrast(image)
                elif func_id == 2:
                    image = self.rand_saturation(image)
            img=image

        return img

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
class RandomBrightness(object):
    def __init__(self, brightness=0.4):
        assert brightness >= 0.0
        assert brightness <= 1.0
        self.brightness = brightness

    def __call__(self, img):
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        return img


class RandomContrast(object):
    def __init__(self, contrast=0.4):
        assert contrast >= 0.0
        assert contrast <= 1.0
        self.contrast = contrast

    def __call__(self, img):
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        return img


class RandomSaturation(object):
    def __init__(self, saturation=0.4):
        assert saturation >= 0.0
        assert saturation <= 1.0
        self.saturation = saturation

    def __call__(self, img):
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        return img




class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.asarray(target).copy(), dtype=torch.int64)
        return image, target

class Resize(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, image, target, ref, percent=None):
        image = F.resize(image, (self.h, self.w))
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        target = F.resize(target, (self.h, self.w), interpolation=Image.NEAREST)
        return image, target, ref, percent

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, ref, percent):
        for t in self.transforms:
            image, target, ref, percnet = t(image, target, ref, percent)
        return image, target, ref, percnet

import collections
class RandomResize(object):
    def __init__(self, base_size, ratio_range, scale=True, bigger_side_to_base_size=True):
        assert isinstance(ratio_range, collections.Iterable) and len(ratio_range) == 2
        self.base_size = base_size
        self.ratio_range = ratio_range
        self.scale = scale
        self.bigger_side_to_base_size = bigger_side_to_base_size

    def __call__(self, img, mask, ref, percent=None):
        img = img
        mask = mask
        w, h = img.size

        if isinstance(self.base_size, int):
            # obtain long_side
            if self.scale:
                long_side = random.randint(int(self.base_size * self.ratio_range[0]),
                                           int(self.base_size * self.ratio_range[1]))
            else:
                long_side = self.base_size

            # obtain new oh, ow
            if self.bigger_side_to_base_size:                                   #  根据长边进行调整
                ratio = float(long_side / float(max(h, w)))
                new_w, new_h = round(w * ratio), round(h * ratio)
            else:
                ratio = float(long_side / float(min(h, w)))
                new_w, new_h = round(w * ratio), round(h * ratio)

            resized_img = img.resize((new_w, new_h), Image.BILINEAR)
            resized_mask = mask.resize((new_w, new_h), Image.NEAREST)
            return resized_img, resized_mask, ref, percent

        elif (isinstance(self.base_size, list) or isinstance(self.base_size, tuple)) and len(self.base_size) == 2:
            if self.scale:
                # scale = random.random() * 1.5 + 0.5  # Scaling between [0.5, 2]
                ratio = self.ratio_range[0] + random.random() * (self.ratio_range[1] - self.ratio_range[0])
                # print("="*100, h, self.base_size[0])
                # print("="*100, w, self.base_size[1])
                new_w, new_h = int(self.base_size[0] * ratio), int(self.base_size[1] * ratio)
            else:
                new_w, new_h = self.base_size
            resized_img = img.resize((new_w, new_h), Image.BILINEAR)
            resized_mask = mask.resize((new_w, new_h), Image.NEAREST)
            return resized_img, resized_mask, ref, percent

        else:
            raise ValueError

class crop(object):
    """
        padding的就不涉及crop的问题
    """
    def __init__(self,base_size: int, ignore_value: int = -1, max_try: int = 20):
        self.size = base_size
        self.ignore_value = ignore_value
        self.max_try = max_try
    def __call__(self, img, mask, ref, percent=None):
        w, h = img.size
        padw = self.size - w if w < self.size else 0
        padh = self.size - h if h < self.size else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask_base = mask.copy()
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.ignore_value)
        # Image.fromarray((np.array(mask) * 255).astype(np.uint8)).show()

        num_try = 0
        while num_try < self.max_try:
            num_try += 1
            # a, b = img.size
            region = T.RandomCrop.get_params(img, [self.size, self.size])  # [i, j, target_w, target_h]
            import cv2
            contours, _ = cv2.findContours(np.array(mask_base), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_rect = cv2.minAreaRect(contours[0])
            region_x1, region_y1, region_x2, region_y2 = region[0], region[1], region[0]+region[2], region[1]+region[3]
            box_x1, box_y1, box_x2, box_y2 = min_rect[0][0]-min_rect[1][0]/2, min_rect[0][1]-min_rect[1][1]/2, min_rect[0][0]+min_rect[1][0]/2, min_rect[0][1]+min_rect[1][1]/2
            if box_x1 >= region_x1 and box_y1 >= region_y1 and box_x2 <= region_x2 and box_y2 <= region_y2:
                img = img.crop((region_x1, region_y1, region_x2, region_y2))
                mask = mask.crop((region_x1, region_y1, region_x2, region_y2))
                return img, mask, ref, percent
        # w, h = img.size
        # x = random.randint(0, w - self.size)
        # y = random.randint(0, h - self.size)
        # img = img.crop((x, y, x + self.size, y + self.size))
        # mask = mask.crop((x, y, x + self.size, y + self.size))
        resized_img = img.resize((self.size, self.size))
        resized_mask = mask.resize((self.size, self.size))

        return resized_img, resized_mask, ref, percent

class RandomHorizontalFlip(object):
    def __call__(self, img, mask, ref, percent=None):
        if percent < 0.5: # random.random()
            img = F.hflip(img)
            mask=F.hflip(mask)
            ref = ref.replace('right', '*&^special^&*').replace('left', 'right').replace('*&^special^&*', 'left')  # 水平翻转后

        return img, mask, ref, percent


# # # # # # # # # # # # # # # # # # # # # # # #
# # # 2. Strong Augmentation for image only
# # # # # # # # # # # # # # # # # # # # # # # #

def get_adpative_magnituide(v_min, v_max, confidence, flag_tough_max=True, sigma=None):
    assert 0 <= confidence <= 1
    # mean
    if flag_tough_max:
        var_mu = v_min + (v_max - v_min) * confidence
    else:
        var_mu = v_max - (v_max - v_min) * confidence
        # sigma
    if sigma is None:
        var_sigma = max(var_mu - v_min, v_max - var_mu)
        # var_sigma /=3.0
        var_sigma /= 2.0
    else:
        var_sigma = sigma
    # print("="*10,f"mu:{var_mu}, std:{var_sigma}")
    # truncated norm
    a = (v_min - var_mu) / var_sigma
    b = (v_max - var_mu) / var_sigma
    import scipy.stats as stats
    rv = stats.truncnorm(a, b, loc=var_mu, scale=var_sigma)
    return rv.rvs()

def img_aug_identity(img, modifier=0.99, scale=None, flag_map_random=True, flag_map_gaussian=False):
    return img

def img_aug_autocontrast(img, modifier=0.99, scale=None, flag_map_random=True, flag_map_gaussian=False):
    return ImageOps.autocontrast(img)


def img_aug_equalize(img, modifier=.99, scale=None, flag_map_random=True, flag_map_gaussian=False):
    return ImageOps.equalize(img)

def img_aug_invert(img, modifier=.99, scale=None, flag_map_random=True, flag_map_gaussian=False):
    return ImageOps.invert(img)

def img_aug_blur(img, modifier=0.99, scale=[0.1, 2.0], flag_map_random=True, flag_map_gaussian=False):
    # sigma = np.random.uniform(scale[0], scale[1])
    assert 0 <= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)
    if flag_map_gaussian:
        sigma = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=True)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        sigma = min_v + v
    # print(f"sigma:{sigma}")
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))

def img_aug_contrast(img, modifier=0.99, scale=[0.05, 0.95], flag_map_random=True, flag_map_gaussian=False):
    assert 0 <= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)
    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v = max_v - v
    # v = float(max_v - min_v)*random.random()
    # # print(min_v, max_v, v)
    # v *= modifier
    # v = max_v - v
    # # print(f"final:{v}")
    return ImageEnhance.Contrast(img).enhance(v)

def img_aug_brightness(img, modifier=0.99, scale=[0.15, 0.95], flag_map_random=True, flag_map_gaussian=False):
    assert 0 <= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)
    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v = max_v - v

    # v = float(max_v - min_v)*random.random()
    # # print(min_v, max_v, v)
    # v *= modifier
    # v = max_v - v
    # # print(f"final:{v}")
    return ImageEnhance.Brightness(img).enhance(v)

def img_aug_color(img, modifier=0.99, scale=[0.1, 0.95], flag_map_random=True, flag_map_gaussian=False):
    assert 0 <= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)

    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v = max_v - v

    # v = float(max_v - min_v)*random.random()
    # # print(min_v, max_v, v)
    # v *= modifier
    # v = max_v - v
    # # print(f"final:{v}")
    return ImageEnhance.Color(img).enhance(v)

def img_aug_sharpness(img, modifier=0.99, scale=[0.05, 0.95], flag_map_random=True, flag_map_gaussian=False):
    assert 0 <= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)

    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v = max_v - v

    # v = float(max_v - min_v)*random.random()
    # # print(min_v, max_v, v)
    # v *= modifier
    # v = max_v - v
    # # print(f"final:{v}")

    return ImageEnhance.Sharpness(img).enhance(v)

def img_aug_hue(img, modifier=0.99, scale=[0, 0.5], flag_map_random=True, flag_map_gaussian=False):
    assert 0 <= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)
    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v += min_v

    # v = float(max_v - min_v)*random.random()
    # v *= modifier
    # v += min_v
    if np.random.random() < 0.5:
        hue_factor = -v
    else:
        hue_factor = v
    # print(f"Final-V:{hue_factor}")
    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img
    h, s, v = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")
    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img

def img_aug_posterize(img, modifier=0.99, scale=[4, 8], flag_map_random=True, flag_map_gaussian=False):
    assert 0 <= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()

    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
        v = int(np.ceil(v))
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v = int(np.ceil(v))
        v = max(1, v)
        v = max_v - v

    # # print(min_v, max_v, v)
    # v *= modifier
    # v = int(np.ceil(v))
    # v = max(1, v)
    # v = max_v - v
    # # print(f"final:{v}")
    return ImageOps.posterize(img, v)

def img_aug_solarize(img, modifier=0.99, scale=[1, 256], flag_map_random=True, flag_map_gaussian=False):
    assert 0 <= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)

    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
        v = int(np.ceil(v))
        v = max(1, v)
        v = min(v, 256)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v = int(np.ceil(v))
        v = max_v - v
        v = max(1, v)
        v = min(v, 256)

    # v = float(max_v - min_v)*random.random()
    # # print(min_v, max_v, v)
    # v *= modifier
    # v = int(np.ceil(v))
    # v = max(1, v)
    # v = max_v - v
    # # print(f"final:{v}")
    return ImageOps.solarize(img, v)

def augment_list():
    l = [
        (img_aug_identity, None),                                           # 返回原始的图像
        (img_aug_autocontrast, None),                                       # 自动对比度
        (img_aug_equalize, None),                                           # 直方图均衡化
        #(img_aug_invert, None),  # hard comment for 10 augs                 反转像素，产生底片效果
        (img_aug_blur, [0.1, 2.0]),                                         # 高斯模糊
        (img_aug_contrast, [0.1, 0.95]),                                    # 对比度
        (img_aug_brightness, [0.1, 0.95]),                                  # 亮度
        (img_aug_color, [0.1, 0.95]),                                       # 颜色
        (img_aug_sharpness, [0.1, 0.95]),                                   # 清晰度
        (img_aug_posterize, [4, 8]),                                        # 颜色值降低到较低的位数范围
        #(img_aug_solarize, [1, 256]), # hard comment for 11/10 augs         # 超过特定阈值的像素进行反转
        #(img_aug_hue, [0, 0.5])                                             # 抖动图像的色调
    ]
    return l

class strong_img_aug:
    def __init__(self, num_augs, flag_map_random=True, flag_map_gaussian=False):
        self.n = num_augs
        self.augment_list = augment_list()
        self.flag_map_random = flag_map_random
        self.flag_map_gaussian = flag_map_gaussian

    def __call__(self, img, modifier=0.999):
        ops = random.choices(self.augment_list, k=self.n)
        for op, scales in ops:
            # print("="*20, str(op))
            img = op(img, modifier, scales, self.flag_map_random, self.flag_map_gaussian)
        return img