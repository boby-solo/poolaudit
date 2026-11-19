"""
torchvision.transforms 的最小等价实现（仅覆盖 CLIP _transform 用到的 4 个算子）。
避免在裸 torch 环境安装 torchvision（防止 pip 连带升级 torch）。
语义对齐 torchvision 0.17：Resize(int) 短边缩放保持长宽比，bicubic 插值。
"""

import torch
from PIL import Image


class _Resize:
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interp = interpolation

    def __call__(self, img):
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            ow = self.size
            oh = int(round(self.size * h / w))
        else:
            oh = self.size
            ow = int(round(self.size * w / h))
        return img.resize((ow, oh), self.interp)


class _CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        left = (w - self.size) // 2
        top = (h - self.size) // 2
        return img.crop((left, top, left + self.size, top + self.size))


class _ToTensor:
    def __call__(self, img):
        arr = torch.from_numpy(__import__("numpy").asarray(img, dtype="uint8").copy())
        return arr.permute(2, 0, 1).float().div(255.0)


class _Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, t):
        return (t - self.mean) / self.std


class Compose:
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x


def Resize(size, interpolation=Image.BICUBIC):
    return _Resize(size, interpolation)


def CenterCrop(size):
    return _CenterCrop(size)


def ToTensor():
    return _ToTensor()


def Normalize(mean, std):
    return _Normalize(mean, std)
