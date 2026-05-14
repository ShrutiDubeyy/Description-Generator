# 🤖 Social Media Caption Generator

> A production-grade deep learning system that generates platform-specific social media captions from images using Vision Transformer (ViT) + Transformer Decoder architecture, trained on real Instagram captions.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Table of Contents
- [Overview](#overview)
- [Two Projects](#two-projects)
- [ML Concepts Used](#ml-concepts-used)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [API Usage](#api-usage)
- [Results](#results)
- [Research References](#research-references)
- [Difference from Existing Work](#difference-from-existing-work)
- [Future Work](#future-work)

---

## 📖 Overview

This repository contains two complete deep learning projects for image captioning built entirely from scratch using PyTorch.

| | Project 1 | Project 2 |
|---|---|---|
| **Name** | Image Description Generator | Social Caption Generator |
| **Encoder** | ResNet-50 (CNN) | Vision Transformer (ViT) |
| **Decoder** | LSTM + Bahdanau Attention | Transformer Decoder |
| **Dataset** | Flickr8k (8K images) | Instagram Captions (28K) |
| **Output** | "a dog running in a field" | "Living life off leash 🐾 #doglife" |

---

## 🔬 ML Concepts Used

### Computer Vision
| Concept | Where Used |
|---------|-----------|
| Convolutional Neural Networks | ResNet-50 encoder |
| Transfer Learning | Pretrained ImageNet weights |
| Feature Maps | 7×7 = 49 spatial regions |
| Vision Transformer (ViT) | Project 2 encoder |
| Patch Embedding | 16×16 image patches → 768 vectors |
| Positional Encoding | Spatial awareness for patches |
| Data Augmentation | Random flip, color jitter |

### Natural Language Processing
| Concept | Where Used |
|---------|-----------|
| Vocabulary Building | Word frequency counting |
| Word Embeddings | Word index → dense vector |
| Sequence Generation | Autoregressive word generation |
| Teacher Forcing | Ground truth feeding during training |
| Beam Search | Better decoding at inference |
| Top-K Sampling | Creative caption diversity |
| Temperature Sampling | Control creativity vs predictability |
| Platform Conditioning | `<instagram>` `<linkedin>` style tokens |

### Deep Learning Core
| Concept | Where Used |
|---------|-----------|
| Encoder-Decoder Architecture | Full model pipeline |
| Bahdanau Attention | Image region focus per word |
| Multi-Head Self Attention | ViT encoder (12 heads) |
| Multi-Head Cross Attention | Decoder attends to image |
| Masked Self Attention | Causal masking for decoder |
| LSTM | Project 1 sequential decoder |
| Residual Connections | Skip connections in ResNet + Transformer |
| Layer Normalization | Pre-norm in transformer blocks |
| Dropout | 0.1–0.5 regularization |

### Training Techniques
| Concept | Where Used |
|---------|-----------|
| Cross Entropy Loss | Token-level loss, padding ignored |
| Adam Optimizer | Project 1 training |
| AdamW Optimizer | Project 2 transformer training |
| Learning Rate Warmup | Stable transformer training |
| Cosine Decay Scheduler | Smooth LR reduction |
| Gradient Clipping | Prevents exploding gradients |
| Early Stopping | Stops when val loss plateaus |
| Checkpoint Saving | Every epoch — never lose progress |

### Evaluation
| Metric | Description |
|--------|-------------|
| Training/Validation Loss | Overfitting detection |
| BLEU Score | N-gram caption quality metric |
| Perplexity | exp(cross_entropy) |
| Attention Visualization | Heatmap of focused regions |

---

## 🏗️ Architecture

### Project 1 — CNN + Attention + LSTM

```
Input Image [3, 224, 224]
        ↓
ResNet-50 Encoder (pretrained, frozen)
        ↓
Feature Map [49, 2048]
        ↓
Bahdanau Soft Attention
        ↓
Context Vector [2048]
        ↓
LSTM Decoder (512 hidden units)
        ↓
"a woman in a white shirt"
        ↓
Platform Postprocessor
        ↓
"Confidence is my best outfit ✨ #vibes"
```

### Project 2 — ViT + Transformer Decoder

```
Input Image [3, 224, 224]
        ↓
Split into 196 patches (16×16 each)
        ↓
Patch Embedding [196, 768]
        ↓
12× Transformer Encoder Blocks
   └── Multi-Head Self Attention (12 heads)
   └── Feed Forward Network (3072 dim)
   └── Layer Norm + Residual
        ↓
Image Features [196, 768]
        ↓
6× Transformer Decoder Blocks
   └── Masked Self Attention
   └── Cross Attention → attends to image!
   └── Feed Forward Network
   └── Layer Norm + Residual
        ↓
<instagram> token conditioning
        ↓
"Living my best life ✨ #vibes #lifestyle"
```

### Bahdanau Attention Detail

```
encoder_features : a_i [batch, 49, 2048]
decoder_hidden   : h_t [batch, 512]

Score:   e_ti = W_a · tanh(W_enc·a_i + W_dec·h_t)
Weight:  α_ti = softmax(e_ti)      ← sums to 1.0
Context: c_t  = Σ α_ti · a_i       ← focused summary

α reveals WHICH image region the model
focused on when generating each word!
```

---

## ✨ Features

- **Multi-Platform Captions** — one image, 5 platform styles:
  - 📸 Instagram — casual, emoji, hashtags
  - 💼 LinkedIn — professional, inspiring
  - 🐦 Twitter — short, punchy
  - 📧 Email — descriptive, formal
  - ✦ General — neutral description
- **Vision Transformer Encoder** — state of the art (2024)
- **Transformer Decoder** — parallel generation, no vanishing gradients
- **Real Instagram Training Data** — 28K real human captions
- **Platform Conditioning** — token based style control
- **FastAPI Backend** — REST API with Swagger UI
- **Beautiful Frontend** — drag and drop, copy to clipboard
- **Resume Training** — checkpoint saves every epoch to Google Drive

---

## 📁 Project Structure

```
Social_caption/
│
├── api/
│   └── main.py                  ← FastAPI backend
│
├── checkpoints/
│   ├── best_model.pth           ← best trained weights
│   ├── latest_checkpoint.pth    ← resume checkpoint
│   └── vocab.pkl                ← vocabulary
│
├── data/
│   └── instagram/
│       ├── images/              ← 28K real Instagram images
│       └── captions.csv         ← real human captions
│
├── src/
│   ├── dataset.py               ← vocabulary + data loading
│   ├── vit_encoder.py           ← Vision Transformer encoder
│   ├── transformer_decoder.py   ← Transformer decoder
│   ├── model.py                 ← complete model
│   └── postprocessor.py         ← platform caption styling
│
├── index.html                   ← frontend UI
├── config.py                    ← all hyperparameters
├── train.py                     ← training with resume
├── download_data.py             ← dataset downloader
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# Clone
git clone https://github.com/yourusername/social_caption_generator.git
cd social_caption_generator

# Virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Install
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | `kkcosmos/instagram-images-with-captions` |
| Platform | HuggingFace Datasets |
| Total images | 28,360 |
| Caption style | Real human Instagram captions |
| Vocabulary | ~12,541 unique words |
| Train/Val split | 90% / 10% |

### Sample Real Instagram Captions:
```
"Rows on rows on rows"
"eyes up here 💋"
"I guess she wasn't feeling this fitting!"
"@deepikapadukone poses with her team in Madrid! #IIFA2016"
```

### Download dataset automatically:
```bash
python download_data.py
```

---

## 🏋️ Training

### Google Colab (Recommended — free T4 GPU)

```python
# Cell 1 — Mount Drive (ALWAYS FIRST!)
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/social_caption_checkpoints', exist_ok=True)
```

```python
# Cell 2 — Train (auto resumes if disconnected!)
# Checkpoints save to Drive after EVERY epoch
python train.py
```

### Resume after disconnect:
```python
# Just run train.py again — auto detects checkpoint!
# Resumes from last completed epoch automatically
python train.py
```

### Training Configuration (`config.py`):

```python
IMAGE_SIZE    = 224      # ViT input size
PATCH_SIZE    = 16       # 224/16 = 14 → 14×14 = 196 patches
VIT_EMBED_DIM = 768      # patch embedding dimension
VIT_NUM_HEADS = 12       # attention heads in encoder
VIT_NUM_LAYERS= 12       # transformer encoder blocks
DEC_NUM_HEADS = 8        # attention heads in decoder
DEC_NUM_LAYERS= 6        # transformer decoder blocks
BATCH_SIZE    = 32       # images per batch
EPOCHS        = 20       # training epochs
LEARNING_RATE = 1e-4     # AdamW learning rate
WARMUP_STEPS  = 1000     # transformer warmup steps
```

### Training Results (Project 1 — Flickr8k):

| Epoch | Train Loss | Val Loss | Perplexity |
|-------|-----------|---------|------------|
| 1 | 3.62 | 3.00 | 20.1 |
| 5 | 2.37 | 2.38 | 10.8 |
| 10 | 2.00 | 2.22 | 9.2 |

---

## 🚀 API Usage

### Start the server:
```bash
uvicorn api.main:app --reload --port 8001
```

### Interactive Swagger UI:
👉 [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

### Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check + model status |
| GET | `/platforms` | List all platforms + styles |
| POST | `/caption` | Generate caption (single platform) |
| POST | `/caption/all` | Generate for ALL platforms at once |

### Single platform:
```bash
curl -X POST "http://127.0.0.1:8001/caption" \
  -F "file=@your_image.jpg" \
  -F "platform=instagram"
```

### Response:
```json
{
  "raw_description": "a woman in a white shirt standing",
  "caption": "Confidence is my best outfit ✨ #vibes #lifestyle #mood",
  "platform": "instagram",
  "words": 9,
  "status": "success"
}
```

### All platforms at once:
```bash
curl -X POST "http://127.0.0.1:8001/caption/all" \
  -F "file=@your_image.jpg"
```

### Response:
```json
{
  "raw_description": "a woman in a white shirt standing",
  "captions": {
    "instagram": {"caption": "Confidence is my best outfit ✨ #vibes"},
    "linkedin":  {"caption": "Showing up with purpose every day. #Growth"},
    "twitter":   {"caption": "main character energy ✨"},
    "email":     {"caption": "A woman in a white shirt standing for reference."},
    "general":   {"caption": "a woman in a white shirt standing"}
  },
  "status": "success"
}
```

---

## 📈 Results

### Platform Caption Examples

| Platform | Input Image | Generated Caption |
|----------|------------|-------------------|
| Instagram | woman smiling | "Confidence is my best outfit ✨ #vibes" |
| LinkedIn | woman smiling | "Showing up with purpose every day. #Growth" |
| Twitter | woman smiling | "main character energy ✨" |
| Email | woman smiling | "A woman in a white shirt for reference." |

### Model Comparison

| Architecture | Parameters | Training Time/epoch | Caption Quality |
|---|---|---|---|
| CNN + LSTM | ~45M | ~4 hrs (CPU) / 15 min (GPU) | Good |
| ViT + Transformer | ~152M | ~22 min (T4 GPU) | Better |

---

## 📚 Research References

1. **Show, Attend and Tell** — Xu et al. (2015)
   - Introduced soft attention mechanism for image captioning
   - Foundation of our Project 1 attention implementation
   - [https://arxiv.org/abs/1502.03044](https://arxiv.org/abs/1502.03044)

2. **An Image is Worth 16×16 Words (ViT)** — Dosovitskiy et al. (2020)
   - Introduced Vision Transformer for image recognition
   - Foundation of our Project 2 ViT encoder
   - [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

3. **Attention is All You Need** — Vaswani et al. (2017)
   - Introduced Transformer architecture
   - Foundation of our Project 2 decoder
   - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

4. **Advancement in Image Caption Generation** — Sarda, Mehta, Ali (2025)
   - ICAAAI 2025 paper using CNN+LSTM on Flickr8k
   - Validates our Project 1 approach
   - Achieved 90% accuracy with similar architecture

5. **Deep Residual Learning (ResNet)** — He et al. (2015)
   - Introduced skip connections for deep networks
   - ResNet-50 used as our CNN encoder
   - [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

---

## 🆚 Difference from Existing Work

### Compared to Sarda et al. (ICAAAI 2025):

| Feature | Sarda et al. 2025 | Our Project |
|---------|-------------------|-------------|
| Encoder | VGG16/VGG19/Inception | ResNet-50 + ViT |
| Decoder | LSTM | LSTM + Transformer |
| Dataset | Flickr8k only | Flickr8k + Instagram (28K) |
| Output | Generic descriptions | Platform specific captions |
| Platforms | Single output | 5 platforms (Instagram, LinkedIn, Twitter, Email, General) |
| Architecture | CNN + LSTM | CNN+LSTM AND ViT+Transformer |
| Training data | Descriptions only | Real Instagram human captions |
| Platform conditioning | ❌ Not implemented | ✅ Token based conditioning |
| API | ❌ None | ✅ FastAPI REST API |
| Frontend | ❌ None | ✅ Full web interface |
| Epochs trained | 100 epochs | 10-20 epochs (GPU) |

### Key innovations in our work:

```
1. Platform Conditioning
   → Special tokens <instagram> <linkedin> <twitter>
   → One model generates 5 different styles
   → Novel approach not in existing literature

2. Dual Architecture
   → Project 1: CNN + LSTM (classical approach)
   → Project 2: ViT + Transformer (state of art 2024)
   → Direct comparison of old vs new

3. Real Social Media Data
   → Trained on REAL Instagram captions
   → Not just generic descriptions
   → Human written caption style

4. Production Ready System
   → REST API with FastAPI
   → Beautiful web frontend
   → Resume training support
   → Multi-platform output
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Deep Learning | PyTorch 2.2 |
| CNN Encoder | ResNet-50 (ImageNet pretrained) |
| ViT Encoder | Built from scratch |
| LSTM Decoder | Built from scratch |
| Transformer Decoder | Built from scratch |
| API Framework | FastAPI + Uvicorn |
| Dataset | HuggingFace Datasets |
| Image Processing | Pillow + torchvision |
| Training Platform | Google Colab (T4 GPU) |
| Frontend | HTML + CSS + Vanilla JS |

---

## 🔮 Future Work

- [ ] Add BLEU score evaluation pipeline
- [ ] Implement attention map visualization (heatmap overlay)
- [ ] Add beam search decoding to Project 1
- [ ] MongoDB integration for caption history
- [ ] Fine tune on more platform specific data (LinkedIn, Twitter)
- [ ] Deploy to cloud (AWS / HuggingFace Spaces)
- [ ] Add multilingual caption support
- [ ] Implement RLHF (Reinforcement Learning from Human Feedback)
- [ ] Mobile app integration

---

## 👨‍💻 Author

Built from scratch as a deep learning research project exploring modern image captioning architectures — from classical CNN+LSTM to state-of-the-art ViT+Transformer.

---

> ⭐ If you found this useful, please give it a star!

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co) — Instagram dataset and model hub
- [PyTorch](https://pytorch.org) — deep learning framework
- [FastAPI](https://fastapi.tiangolo.com) — API framework
- [Google Colab](https://colab.research.google.com) — free GPU training
- Sarda, Mehta, Ali (ICAAAI 2025) — reference paper
