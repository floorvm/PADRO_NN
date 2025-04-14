from torchvision.datasets import MNIST
from typing import Any
from PIL import Image
import torch
import numpy as np
from scipy.signal import convolve2d


def convolve(H, x):
    return convolve2d(x.squeeze(), H, mode='same',boundary='fill', fillvalue=0)


class MNISTMaskDataset(MNIST):
    def __init__(self, rootPath, train=True, download=True, transform=None) -> None:
        super().__init__(rootPath, train, transform, download)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, targetIdx = self.data[index], int(self.targets[index])
        
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)
        
        maskImg = torch.repeat_interleave(torch.zeros_like(img, dtype=torch.float32), 10, 0)
        maskImg[targetIdx] = img != 0

        return img, maskImg, targetIdx


class MNISTBlurredDataset(MNIST):
    def __init__(self, H, rootPath, train=True, download=True, transform=None) -> None:
        print(f"Initializing MNISTBlurredDataset with rootPath: {rootPath}")
        self.data = MNIST(root=rootPath, train=train, download=download, transform=transform)
        self.H = H
        self.transform = transform


    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, _ = self.data[index]
        # print(f"Original Type of image: {type(img)}")
        # print(f"Original Shape of image: {img.shape if hasattr(img, 'shape') else 'No shape attribute'}")

        img = (img.numpy().squeeze() * 255).astype(np.uint8)
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        # print(f"Post-tranform Type of image: {type(img)}")
        # print(f"Post-tranform Shape of image: {img.shape if hasattr(img, 'shape') else 'No shape attribute'}")

        img_blurred = torch.tensor(convolve(self.H, img)).unsqueeze(0)

        return img, img_blurred