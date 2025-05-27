from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import numpy as np


def convolve(H, x):
    return convolve2d(x.squeeze(), H, mode='same',boundary='fill', fillvalue=0)


def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def isscalar(tensor):
    return tensor.dim() == 0


def plot_images(trainLoader, num_images=5):
    data_iter = iter(trainLoader)
    blurred_images, images = next(data_iter)

    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(10, 2 * num_images))

    for i in range(num_images):
        ax = axes[i, 0]
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'Original Image {i+1}')
        ax.axis('off')

        ax = axes[i, 1]
        ax.imshow(blurred_images[i].squeeze(), cmap='gray')
        ax.set_title(f'Blurred Image {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_predictions(images, labels, predictions, num=1):

    images = images.detach().cpu()
    labels = labels.detach().cpu()
    predictions = predictions.detach().cpu()

    num = min(num, images.size(0))

    plt.figure(figsize=(6, num * 2))
    for i in range(num):
        plt.subplot(num, 3, i * 3 + 1)
        plt.imshow(images[i, 0, :, :], cmap='gray')
        plt.title('Blurred')
        plt.axis('off')

        plt.subplot(num, 3, i * 3 + 2)
        plt.imshow(labels[i, 0, :, :], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(num, 3, i * 3 + 3)
        plt.imshow(predictions[i, 0, :, :], cmap='gray')
        plt.title('Deblurred')
        plt.axis('off')

    plt.tight_layout()
    plt.show()