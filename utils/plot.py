import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import measure
from skimage.color import gray2rgb
from skimage.draw import polygon_perimeter
from utils.util import isSquareRootInteger


def generateColorByClass(numClass: int = 10) -> list[tuple[int, int, int]]:
    if numClass > 256 * 256 * 256:
        raise ValueError("Too many classes for unique RGB generation")

    colors = set()
    while len(colors) < numClass:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.add(color)

    return list(colors)


def getDrawingMaskOnImage(mask: np.ndarray, image: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = gray2rgb(image)  # Convert grayscale to RGB

    contours = measure.find_contours(mask, 0.5)
    for contour in contours:
        rr, cc = polygon_perimeter(contour[:, 0], contour[:, 1], shape=image.shape[:2], clip=True)
        image[rr, cc] = color  # Apply contour color

    colorMask = np.zeros_like(image)
    colorMask[mask == 255] = color
    result = (image * 0.85 + colorMask * 0.15).astype(np.uint8)
    return result


def getDrawingTitleImage(title: str, image: np.ndarray, textColor: tuple[int, int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.load_default()  # Use default font
    except:
        font = None  # If no font available, use default

    draw.text((0, 6), title, fill=textColor, font=font)
    return np.array(pil_image)


def saveImagesForMnistSegment(images, labelsByClass, colorByClass, threshold, savePath,
                              fileName='segmentImage.jpg'):
    imageNum = len(images)
    if isSquareRootInteger(imageNum):
        imageRowNum = int(np.sqrt(imageNum))
    else:
        raise ValueError("Number of images is not appropriate.")

    imageSize = images[0].shape[1]
    saveImage = np.zeros((imageSize * imageRowNum, imageSize * imageRowNum, 3), dtype=np.uint8)

    for idx, image in enumerate(images):
        image = image.permute(1, 2, 0).mul(255).byte().cpu().numpy()
        labelByClass = labelsByClass[idx]

        for classIdx, label in enumerate(labelByClass):
            label[label > threshold] = 255
            label[label <= threshold] = 0
            label = label.byte().cpu().numpy()
            image = getDrawingMaskOnImage(label, image, colorByClass[classIdx])

        i, j = idx % imageRowNum, idx // imageRowNum
        saveImage[j * imageSize:j * imageSize + imageSize, i * imageSize:i * imageSize + imageSize] = image

    pil_save_image = Image.fromarray(saveImage)
    pil_save_image.save(os.path.join(savePath, fileName))
