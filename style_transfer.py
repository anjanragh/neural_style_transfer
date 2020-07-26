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

import math
import streamlit as st

st.title("Neural Style Transfer Demo!")
st.set_option('deprecation.showfileUploaderEncoding', False)

# Create device so that it uses cuda if available. If not, just use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 128

# Preprocess the image i.e. Resize and convert into Tensor

loader = transforms.Compose(
    [transforms.Resize((imsize, imsize)), transforms.ToTensor()])


# Helper function to load and process image
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = st.file_uploader("Choose an image...", type="jpg", key="style")
content_img = st.file_uploader("Choose an image...", type="jpg", key="content")
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("")
#     st.write("Classifying...")
#     label = predict(uploaded_file)
#     st.write('%s (%.2f%%)' % (label[1], label[2]*100))

# Store info about the images. We need two images, one for style and the other for content.
image_dir = Path("./images")
if style_img is not None and content_img is not None:
    style_image = image_loader(style_img)
    content_image = image_loader(content_img)

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
        st.image(image, width=256)
        if title is not None:
            plt.title(title)
        plt.pause(0.01)

    imshow(style_image)
    imshow(content_image)

    class ContentLoss(nn.Module):
        def __init__(self, target):
            super(ContentLoss, self).__init__()
            # We detach it because the loss calculation step will throw errors otherwise.
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
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

    # VGG normalized for mean and std from ImageNet. This is a configuration often used for image normalization.
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

    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_image, content_image,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):

        cnn = copy.deepcopy(cnn)

        # Normalization Module
        normalization = Normalization(
            normalization_mean, normalization_std).to(device)

        # To store the losses
        content_losses = []
        style_losses = []

        # Add normalization layer to the cnn. This assumes that the cnn is created by nn.Sequential
        model = nn.Sequential(normalization)

        i = 0  # We increment everytime we see a conv layer
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)

            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)

            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)

            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)

            else:
                raise RuntimeError('Unrecognized Layer: {}'.format(
                    layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # Add contentLoss
                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # Add style loss
                target_feature = model(style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # Now we don't care about the layers after the style and content losses. So we remove them
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:i+1]

        return model, style_losses, content_losses

    input_image = content_image.clone()

    def get_input_optimizer(input_image):
        optimizer = optim.LBFGS([input_image.requires_grad_()])
        return optimizer

    def run_style_transfer(model, style_losses, content_losses, input_image,
                           num_steps=300, style_weight=1000000, content_weight=1):

        optimizer = get_input_optimizer(input_image)

        print("Optimizing...")
        # Add a placeholder
        latest_iteration = st.empty()
        bar = st.progress(0)

        run = [0]

        while(run[0] <= num_steps):
            val = math.floor((run[0]//num_steps)*100)
            latest_iteration.text(f'Optimizing {val}')
            bar.progress(val)

            def closure():
                input_image.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_image)

                style_score = 0
                content_score = 0

                for style_layer in style_losses:
                    style_score += (1/5)*style_layer.loss

                for content_layer in content_losses:
                    content_score += content_layer.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1

                if run[0] % 50 == 0:
                    print("run {}:".format(run[0]))
                    print("Style loss : {:4f}, Content Loss : {:4f}".format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score+content_score

            optimizer.step(closure)

        input_image.data.clamp_(0, 1)

        return input_image

    print("Building the style transfer model...")

    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, cnn_normalization_mean, cnn_normalization_std, style_image, content_image)

    output = run_style_transfer(
        model, style_losses, content_losses, input_image, num_steps=300)

    imshow(output, title="Final image")
