import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
import math
import numpy as np
from utils.util import createIncrementalPath
from datasets.MNISTMaskDataset import MNISTBlurredDataset, MNISTMaskDataset
from model.unet import UNet
from utils.plot2 import saveImagesForMnistSegment, generateColorByClass
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    """ Generate Gaussian kernel

    Parameters:
    :param size: Size of kernel
    :param sigma: Standard deviation of Gaussian
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def plot_images(trainLoader, num_images=5):
    # Get a batch of images and their corresponding blurred versions
    data_iter = iter(trainLoader)
    images, blurred_images = next(data_iter)
    print(images[0])

    # We will display 'num_images' images
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(10, 2 * num_images))

    for i in range(num_images):
        # Non-blurred image (original)
        ax = axes[i, 0]
        ax.imshow(images[i].squeeze(), cmap='gray')  # Convert from tensor to image
        ax.set_title(f'Original Image {i+1}')
        ax.axis('off')

        # Blurred image
        ax = axes[i, 1]
        ax.imshow(blurred_images[i].squeeze(), cmap='gray')  # Convert from tensor to image
        ax.set_title(f'Blurred Image {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

torch.autograd.set_detect_anomaly(True)

batchSize = 64
imgSize = 28
labelNum = 10
threshold = .8
plot = True
colorByClass = generateColorByClass(labelNum)

learningRate = 1e-3
numEpoch = 100

H = torch.tensor(gaussian_kernel(3, sigma=0.5))

resultPath = createIncrementalPath('./result')
path = "./data/MNIST"
os.makedirs(path, exist_ok=True)
transform = transforms.Compose([transforms.Resize(imgSize), transforms.ToTensor()])
trainDataset = MNISTBlurredDataset(H, path, train=True, download=True, transform=transform)
trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True)
trainLoaderSize = len(trainLoader)
# for i, (images, labels) in enumerate(trainLoader):
#     print(images.shape)
#     print(labels.shape)
plot_images(trainLoader, num_images=2)

testDataset = MNISTBlurredDataset(H, path, train=False, download=True, transform=transform)
testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=True, drop_last=True)
testLoaderSize = len(testLoader)

unet = UNet(2).cpu()
optimizer = torch.optim.Adam(unet.parameters(), lr=learningRate, amsgrad=True)
bceLossFunc = nn.MSELoss()

print("Train Start!")
bestLoss = math.inf
for epoch in range(numEpoch):
    unet.train()
    totalLoss = 0.
    for i, (images, labels) in enumerate(trainLoader):
        images = images.cpu()
        labels = labels.cpu()
        predict = unet(images)

        loss = bceLossFunc(predict, labels)
        totalLoss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    trainLoss = totalLoss / trainLoaderSize

    with torch.no_grad():
        unet.eval()
        totalLoss = 0.
        if plot:
            images, labels, numbers = next(iter(testLoader))
            predict = unet(images.cpu()).sigmoid()
            saveImagesForMnistSegment(images, predict, numbers, colorByClass, threshold, resultPath,
                                      f'epoch{epoch}.jpg')
            for images, labels, numbers in testLoader:
                images = images.cpu()
                labels = labels.cpu()
                predict = unet(images)

                loss = bceLossFunc(predict, labels)
                totalLoss += loss.item()

        else:
            for images, labels, numbers in testLoader:
                images = images.cpu()
                labels = labels.cpu()
                predict = unet(images)

                loss = bceLossFunc(predict, labels)
                totalLoss += loss.item()

    if totalLoss < bestLoss:
        torch.save(unet.state_dict(), os.path.join(resultPath, 'best.pt'))
        bestLoss = totalLoss
    print(f"epoch: {epoch + 1}, Train Loss: {trainLoss}, Test Loss: {totalLoss / testLoaderSize}")