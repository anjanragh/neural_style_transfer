from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
from pathlib import Path

import numpy as np

# Create device so that it uses cuda if available. If not, just use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 128

# Preprocess the image i.e. Resize and convert into Tensor

loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])


# Helper function to load and process image
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# Store info about the images. We need two images, one for style and the other for content.
image_dir = Path("./images")
style_image = image_loader(image_dir/'pixelart.jpg')
content_image = image_loader(image_dir/'anjan.JPG')

assert style_image.size() == content_image.size()




