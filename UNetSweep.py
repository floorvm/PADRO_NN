import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
import math
import numpy as np
from utils.util import createIncrementalPath
from datasets.MNISTMaskDataset import MNISTMaskDataset, MNISTBlurredDataset
from model.unet import UNet
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import random
from scipy.signal import convolve2d
from torch.autograd import Variable
from utils.helpers import plot_predictions, plot_images, isscalar, gaussian_kernel, convolve
from utils.lossfunction import PADRO_Loss
import wandb

wandb.login()

def adjust_lr(optimizer, lr0, epoch):
    """ Adjust learning rate according to epoch. """
    lr = lr0 * (1.0 / np.sqrt(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(config=None):
    with wandb.init(config=config, project="PADRO-UNet"):
        config = wandb.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.autograd.set_detect_anomaly(True)

        batchSize = 32
        imgSize = 28
        plot = True

        learningRateUNet = config.learningRateUNet
        learningRateLamb = config.learningRateLamb
        delta = config.delta
        Lambda = torch.nn.Parameter(torch.tensor(config.Lambda, device=device))
        numEpoch = 100
        K_sample_max = 5
        eps = 0.001

        H = torch.tensor(gaussian_kernel(3, sigma=1))

        path = "./data"
        os.makedirs(path, exist_ok=True)
        transform = transforms.Compose([transforms.Resize(imgSize), transforms.ToTensor()])

        trainDataset = MNISTBlurredDataset(H, path, train=True, download=True, transform=transform)
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True)
        trainLoaderSize = len(trainLoader)

        testDataset = MNISTBlurredDataset(H, path, train=False, download=True, transform=transform)
        testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=True, drop_last=True)
        testLoaderSize = len(testLoader)

        # plot_images(trainLoader, 2)
        unet = UNet(2).to(device)

        optimizer = torch.optim.Adam([
            {"params": unet.parameters(), "lr": learningRateUNet},
            {"params": [Lambda], "lr": learningRateLamb}
        ], amsgrad=True)

        resultPath = createIncrementalPath('./result')
        print("Train Start!")
        bestLoss = math.inf

        # Generate distribution for sampling variable K_sample of RT-MLMC estimator
        elements = np.arange(0, K_sample_max)
        probabilities = 0.5 ** elements
        probabilities = probabilities / np.sum(probabilities)

        for epoch in range(numEpoch):
            unet.train()
            totalLoss = 0.
            for i, (images, labels) in enumerate(trainLoader):
                K_sample = int(np.random.choice(list(elements), size=1, p=list(probabilities))[0])
                images = images.float().to(device)
                labels = labels.float().to(device)

                loss = PADRO_Loss(Lambda, unet, H, delta, eps, K_sample, probabilities, labels, images)

                totalLoss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                  Lambda.data.clamp_(min=1e-6)

                del loss
                torch.cuda.empty_cache()

            trainLoss = totalLoss / trainLoaderSize

            with torch.no_grad():
                unet.eval()
                totalLoss = 0.
                if plot:
                    images, labels = next(iter(testLoader))
                    images = images.float().to(device)
                    predict = unet(images).sigmoid()
                    plot_predictions(images, labels, predict)

                    for images, labels in testLoader:
                        K_sample = int(np.random.choice(list(elements), 1, list(probabilities)))
                        images = images.float().to(device)
                        labels = labels.float().to(device)

                        loss = PADRO_Loss(Lambda, unet, H, delta, eps, K_sample, probabilities, labels, images).float()
                        totalLoss += loss.item()

                else:
                    for images, labels in testLoader:
                        images = images.float().to(device)
                        labels = labels.float().to(device)

                        loss = PADRO_Loss(Lambda, unet, H, delta, eps, K_sample, probabilities, labels, images).float()
                        totalLoss += loss.item()

                adjust_lr(optimizer, learningRateUNet, epoch + 1)
                adjust_lr(optimizer, learningRateLamb, epoch + 1)

            del loss
            torch.cuda.empty_cache()

            testLoss = totalLoss / testLoaderSize
            wandb.log({
                'epoch': epoch + 1,
                'Train Loss': trainLoss,
                'Test Loss': testLoss,
                'Lambda': Lambda.item(),
                'delta': delta,
            })

            if totalLoss < bestLoss:
                torch.save(unet.state_dict(), os.path.join(resultPath, 'best.pt'))
                bestLoss = totalLoss
            print(f"epoch: {epoch + 1}, Lambda: {Lambda}, Train Loss: {trainLoss}, Test Loss: {totalLoss / testLoaderSize}")

sweep_configuration = {
    "name": "first-sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "testLoss"},
    "parameters": {
        "Lambda": {'min': 0.1, 'max': 10.0},
        "delta": {'min': 0.1, 'max': 10.0},
        "learningRateUNet": {'min': 1e-5, 'max': 1.0, "distribution": "log_uniform_values"},
        "learningRateLamb": {'min': 1e-5, 'max': 1.0, "distribution": "log_uniform_values"}
    }
}

sweep_id = wandb.sweep(sweep_configuration, project="PADRO-UNet")
wandb.agent(sweep_id, function=main, count=20)
wandb.finish()