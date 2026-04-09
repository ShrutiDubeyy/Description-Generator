# src/encoder.py

import torch
import torch.nn as nn
import torchvision.models as models

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class Encoder(nn.Module):
    # nn.Module is the base class for ALL neural networks in PyTorch
    # every custom model must inherit from it

    def __init__(self):
        super(Encoder, self).__init__()
        # super().__init__() calls the parent class (nn.Module) setup
        # always required when inheriting from nn.Module

        # ── Load pretrained ResNet-50 ─────────────────────────────────────────
        # weights="IMAGENET1K_V1" means load weights already trained on ImageNet
        # this gives us a huge head start — the CNN already knows how to see!
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # ── Remove the last 2 layers ──────────────────────────────────────────
        # ResNet-50 originally ends with:
        #   → AdaptiveAvgPool (squishes 7x7 into 1x1)
        #   → Linear layer   (classifies into 1000 ImageNet categories)
        # We don't want classification — we want the spatial feature map
        # So we strip those last 2 layers off
        modules = list(resnet.children())[:-2]

        # nn.Sequential wraps a list of layers into one callable model
        self.resnet = nn.Sequential(*modules)

        # ── Add a projection layer ────────────────────────────────────────────
        # ResNet outputs 2048 channels — we project down to ENCODER_DIM
        # This is like a "translator" that brings features to a standard size
        # kernel_size=1 means we apply it per-pixel, not across spatial regions
        self.projection = nn.Conv2d(
            in_channels=2048,
            out_channels=config.ENCODER_DIM,
            kernel_size=1
        )

        # ── Batch normalization ───────────────────────────────────────────────
        # normalizes the output of projection layer
        # makes training more stable and faster
        self.bn = nn.BatchNorm2d(config.ENCODER_DIM)

        # ── Freeze ResNet weights ─────────────────────────────────────────────
        # we don't want to retrain ResNet from scratch — it already works great
        # freezing means these weights won't change during our training
        for param in self.resnet.parameters():
            param.requires_grad = False
            # requires_grad = False means "don't calculate gradients for this"
            # gradients are what PyTorch uses to update weights
            # no gradient = no update = frozen weight

    def forward(self, images):
        # forward() defines what happens when data passes through the encoder
        # PyTorch calls this automatically when you do: encoder(images)

        # images shape coming in: [batch_size, 3, 224, 224]
        # 3 = RGB channels, 224x224 = image dimensions

        # ── Pass through frozen ResNet ────────────────────────────────────────
        with torch.no_grad():
            # torch.no_grad() tells PyTorch: don't track gradients here
            # saves memory since ResNet is frozen anyway
            features = self.resnet(images)
            # features shape: [batch_size, 2048, 7, 7]

        # ── Project from 2048 → ENCODER_DIM ──────────────────────────────────
        features = self.projection(features)
        # features shape: [batch_size, ENCODER_DIM, 7, 7]

        # ── Normalize ─────────────────────────────────────────────────────────
        features = self.bn(features)
        # shape stays: [batch_size, ENCODER_DIM, 7, 7]

        # ── Reshape for attention ─────────────────────────────────────────────
        # attention mechanism expects shape: [batch_size, 49, ENCODER_DIM]
        # 49 = 7x7 = number of image regions
        # we're saying: "give me 49 feature vectors, one per region"

        batch_size = features.size(0)

        # permute swaps the dimensions: [batch, channels, h, w] → [batch, h, w, channels]
        features = features.permute(0, 2, 3, 1)
        # shape: [batch_size, 7, 7, ENCODER_DIM]

        # view reshapes the tensor: flatten 7x7 into 49
        features = features.view(batch_size, -1, config.ENCODER_DIM)
        # shape: [batch_size, 49, ENCODER_DIM]
        # -1 tells PyTorch to figure out that dimension automatically (7×7=49)

        return features

    def fine_tune(self, allow=True):
        # call this later when you want to unfreeze ResNet for fine-tuning
        # we start frozen, then optionally unfreeze after initial training
        for param in self.resnet.parameters():
            param.requires_grad = allow
        print(f"ResNet fine-tuning: {'ON' if allow else 'OFF'}")
        

