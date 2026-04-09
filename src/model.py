# src/model.py

import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from src.encoder import Encoder
from src.decoder import Decoder


class CaptionModel(nn.Module):

    def __init__(self, vocab_size):
        super(CaptionModel, self).__init__()

        # create encoder — the eyes of our model
        self.encoder = Encoder()

        # create decoder — the mouth of our model
        # attention is already INSIDE the decoder
        self.decoder = Decoder(vocab_size=vocab_size)

    def forward(self, images, captions):
        """
        This is called during TRAINING.

        images   : [batch_size, 3, 224, 224]  — real photos
        captions : [batch_size, max_len]       — encoded captions
        """

        # Step 1 — pass images through encoder
        # encoder looks at image and returns 49 region features
        encoder_features = self.encoder(images)
        # shape: [batch_size, 49, 2048]

        # Step 2 — pass features + captions through decoder
        # decoder generates word predictions using attention
        predictions, alphas = self.decoder(encoder_features, captions)
        # predictions: [batch_size, caption_len, vocab_size]
        # alphas:      [batch_size, caption_len, 49]

        return predictions, alphas

    def generate_caption(self, image, vocab, platform="<general>"):
        """
        This is called during INFERENCE (real world usage).
        Takes ONE real image and generates a caption for it.

        image    : [1, 3, 224, 224]  — single preprocessed image
        vocab    : our Vocabulary object
        platform : which style to generate for
        """

        # no gradient tracking needed during inference
        # saves memory and makes it faster
        with torch.no_grad():

            # Step 1 — encode the image
            encoder_features = self.encoder(image)
            # shape: [1, 49, 2048]

            # Step 2 — generate caption word by word
            caption, attention_maps = self.decoder.generate(
                encoder_features,
                vocab,
                platform=platform
            )

        return caption, attention_maps

    def save(self, path):
        """
        Saves the entire model weights to a file.
        Call this after training to preserve what the model learned.
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        """
        Loads previously saved weights back into the model.
        Call this when you want to use a trained model.
        """
        self.load_state_dict(torch.load(path, map_location=config.DEVICE))
        print(f"Model loaded from {path}")



        