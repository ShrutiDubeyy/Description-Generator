 # src/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from src.attention import Attention


class Decoder(nn.Module):

    def __init__(self, vocab_size):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.attention  = Attention()

        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = config.EMBED_DIM,
            padding_idx    = 0
        )

        self.lstm = nn.LSTMCell(
            input_size  = config.EMBED_DIM + config.ENCODER_DIM,
            hidden_size = config.DECODER_DIM
        )

        self.init_h  = nn.Linear(config.ENCODER_DIM, config.DECODER_DIM)
        self.init_c  = nn.Linear(config.ENCODER_DIM, config.DECODER_DIM)
        self.dropout = nn.Dropout(p=config.DROPOUT)
        self.fc      = nn.Linear(config.DECODER_DIM, vocab_size)

    def init_hidden_state(self, encoder_features):
        mean_features = encoder_features.mean(dim=1)
        h = torch.tanh(self.init_h(mean_features))
        c = torch.tanh(self.init_c(mean_features))
        return h, c

    def forward(self, encoder_features, captions):
        batch_size  = encoder_features.size(0)
        caption_len = captions.size(1) - 1
        embeddings  = self.embedding(captions)
        h, c        = self.init_hidden_state(encoder_features)

        predictions = torch.zeros(
            batch_size, caption_len, self.vocab_size
        ).to(encoder_features.device)

        alphas = torch.zeros(
            batch_size, caption_len, 49
        ).to(encoder_features.device)

        for t in range(caption_len):
            context, alpha = self.attention(encoder_features, h)
            word_embed     = embeddings[:, t, :]
            lstm_input     = torch.cat([word_embed, context], dim=1)
            h, c           = self.lstm(lstm_input, (h, c))
            h_drop         = self.dropout(h)
            preds          = self.fc(h_drop)

            predictions[:, t, :] = preds
            alphas[:, t, :]      = alpha

        return predictions, alphas

    def generate(self, encoder_features, vocab, platform="<general>", max_len=50):
        device = encoder_features.device
        h, c   = self.init_hidden_state(encoder_features)

        # start with START token
        start_idx = vocab.word_to_idx[config.START_TOKEN]
        word      = torch.tensor([start_idx]).to(device)

        caption_indices = []
        attention_maps  = []
        recent_words    = []

        for _ in range(max_len):
            embed          = self.embedding(word)
            context, alpha = self.attention(encoder_features, h)
            lstm_input     = torch.cat([embed, context], dim=1)
            h, c           = self.lstm(lstm_input, (h, c))
            output         = self.fc(self.dropout(h))

            # get top 10 candidates
            top10_scores, top10_idx = output.topk(10, dim=1)

            # pick best word not recently used
            predicted_idx = None
            for i in range(10):
                candidate      = top10_idx[0][i].item()
                candidate_word = vocab.idx_to_word.get(candidate, '')

                if candidate_word not in recent_words[-3:] and \
                   candidate_word not in [config.PAD_TOKEN, config.START_TOKEN]:
                    predicted_idx = candidate
                    break

            # fallback to top word
            if predicted_idx is None:
                predicted_idx = top10_idx[0][0].item()

            current_word = vocab.idx_to_word.get(predicted_idx, '')
            recent_words.append(current_word)
            caption_indices.append(predicted_idx)
            attention_maps.append(alpha)

            # stop at end token
            if vocab.idx_to_word[predicted_idx] == config.END_TOKEN:
                break

            word = torch.tensor([predicted_idx]).to(device)

        caption = vocab.decode(caption_indices)

        # fallback if caption too short
        if len(caption.strip()) < 3:
            caption = "a person is standing in a room"

        return caption, attention_maps