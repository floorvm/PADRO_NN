# import shutil
# shutil.rmtree("./data/MNIST", ignore_errors=True)
#
# from torchvision import datasets
# datasets.MNIST(root="./data", train=True, download=True)


import torch
print(torch.cuda.is_available())  # This should return True if CUDA is available
