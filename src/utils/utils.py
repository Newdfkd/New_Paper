import torch
import  torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

class Trasform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str(self.transform)


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    def __call__(self, x, reverse=False):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        return normalize(x, mean, std, reverse=reverse)

class NormalizerCifar100(object):
            def __init__(self):
                mean = (0.5071, 0.4867, 0.4408)
                std = (0.2675, 0.2565, 0.2761)

            def __call__(self, x, reverse=False):
                mean = (0.5071, 0.4867, 0.4408)
                std = (0.2675, 0.2565, 0.2761)
                return normalize(x, mean, std, reverse=reverse)


Augmentation = Trasform([
    # Training view
    transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        Normalizer()
    ]),
    # Testing view
    transforms.Compose([
        Normalizer()
    ])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
