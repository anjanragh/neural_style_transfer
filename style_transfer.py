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

# To convert from tensor to image, we need to create an unloader
unloader = transforms.ToPILImage()

plt.ion()

# Helper function to show the tensor as an image
def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.01)

imshow(style_image)
imshow(content_image)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # We detatch it because the loss calculation step will throw errors otherwise. 
        self.target = target.detatch()

    def forward(self, input):
        loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()

    features = input.view(a*b, c*d)

    G = torch.mm(features, features.t())

    return G.div(a*b*c*d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()

        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# Import the model. The paper uses VGG19 which is what I am going to use
cnn = models.vgg19(pretrained=True).features.to(device).eval()

#VGG normalized for mean and std from ImageNet. This is a configuration often used for image normalization.
cnn_normalization_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.Tensor([0.229, 0.224, 0.225]).to(device)


# If we create a class to normalize the images, then its easier to add it to our nn sequential model
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        self.mean = torch.Tensor(mean).view(-1, 1, 1)
        self.std = torch.Tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean)/self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


