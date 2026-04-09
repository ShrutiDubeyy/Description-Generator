# config.py

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR  = os.path.join(ROOT_DIR, "data", "processed")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# ─── Image settings ───────────────────────────────────────────────────────────
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

# ─── Vocabulary settings ──────────────────────────────────────────────────────
VOCAB_SIZE    = 10000
MIN_WORD_FREQ = 5
PAD_TOKEN     = "<pad>"
START_TOKEN   = "<start>"
END_TOKEN     = "<end>"
UNK_TOKEN     = "<unk>"

# ─── Model architecture ───────────────────────────────────────────────────────
ENCODER_DIM   = 2048
ATTENTION_DIM = 512
EMBED_DIM     = 512
DECODER_DIM   = 512
DROPOUT       = 0.5

# ─── Training settings ────────────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 30
LEARNING_RATE   = 4e-4
GRAD_CLIP       = 5.0
MAX_CAPTION_LEN = 50
DEVICE          = "cuda"

# ─── Platform tokens ──────────────────────────────────────────────────────────
PLATFORMS = ["<instagram>", "<linkedin>", "<twitter>", "<email>", "<general>"] 