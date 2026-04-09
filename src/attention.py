# src/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

        # ── Three learned linear layers ───────────────────────────────────────

        # this layer processes the encoder features (image regions)
        # takes each of the 49 regions: ENCODER_DIM → ATTENTION_DIM
        self.encoder_att = nn.Linear(config.ENCODER_DIM, config.ATTENTION_DIM)

        # this layer processes the decoder hidden state (what word came before)
        # DECODER_DIM → ATTENTION_DIM
        self.decoder_att = nn.Linear(config.DECODER_DIM, config.ATTENTION_DIM)

        # this final layer collapses ATTENTION_DIM → 1 score per region
        # gives us one number per region = "how relevant is this region?"
        self.full_att = nn.Linear(config.ATTENTION_DIM, 1)

        # ── Activation functions ──────────────────────────────────────────────

        # ReLU: any negative value becomes 0, positives stay
        # adds non-linearity so model can learn complex patterns
        self.relu = nn.ReLU()

        # softmax: converts raw scores into probabilities that sum to 1
        # e.g. [2.1, 0.3, 1.5, ...] → [0.4, 0.05, 0.2, ...]
        # dim=1 means apply softmax across the 49 regions
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_features, decoder_hidden):
        """
        encoder_features : [batch_size, 49, ENCODER_DIM]  — the 49 image regions
        decoder_hidden   : [batch_size, DECODER_DIM]      — current decoder state
        """

        # ── Step 1: score each image region ──────────────────────────────────

        # process image features through encoder_att layer
        # shape: [batch_size, 49, ENCODER_DIM] → [batch_size, 49, ATTENTION_DIM]
        att_enc = self.encoder_att(encoder_features)

        # process decoder hidden state through decoder_att layer
        # unsqueeze(1) adds a dimension: [batch, DECODER_DIM] → [batch, 1, DECODER_DIM]
        # this lets us broadcast across all 49 regions in the next step
        att_dec = self.decoder_att(decoder_hidden).unsqueeze(1)
        # shape: [batch_size, 1, ATTENTION_DIM]

        # add both together — this is the "Bahdanau" style attention
        # PyTorch automatically broadcasts att_dec across all 49 regions
        # relu adds non-linearity to the combined signal
        combined = self.relu(att_enc + att_dec)
        # shape: [batch_size, 49, ATTENTION_DIM]

        # collapse to one score per region using full_att
        # squeeze(2) removes the last dimension of size 1
        scores = self.full_att(combined).squeeze(2)
        # shape: [batch_size, 49]

        # ── Step 2: convert scores to weights using softmax ───────────────────

        # softmax turns raw scores into probabilities — all 49 weights sum to 1
        # a region with weight 0.4 gets 40% of the attention
        alpha = self.softmax(scores)
        # shape: [batch_size, 49]  ← these are our attention weights

        # ── Step 3: create weighted context vector ────────────────────────────

        # multiply each region's features by its attention weight
        # unsqueeze(2) makes alpha: [batch, 49] → [batch, 49, 1]
        # so it can multiply with encoder_features: [batch, 49, ENCODER_DIM]
        context = (encoder_features * alpha.unsqueeze(2)).sum(dim=1)
        # .sum(dim=1) adds up all 49 weighted regions into one vector
        # context shape: [batch_size, ENCODER_DIM]
        # this is the "what to focus on" vector passed to the decoder

        return context, alpha
        # we return alpha too — useful for visualizing where model looked!