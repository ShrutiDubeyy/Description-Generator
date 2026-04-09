# src/dataset.py

import os
import json
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


# ─── Vocabulary ───────────────────────────────────────────────────────────────

class Vocabulary:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq   = Counter()
        self.idx         = 0

        self._add_word(config.PAD_TOKEN)
        self._add_word(config.START_TOKEN)
        self._add_word(config.END_TOKEN)
        self._add_word(config.UNK_TOKEN)

        for platform in config.PLATFORMS:
            self._add_word(platform)

    def _add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.idx
            self.idx_to_word[self.idx] = word
            self.idx += 1

    def build(self, captions):
        for caption in captions:
            self.word_freq.update(caption.lower().split())

        for word, freq in self.word_freq.items():
            if freq >= config.MIN_WORD_FREQ:
                self._add_word(word)

    def __len__(self):
        return len(self.word_to_idx)

    def encode(self, caption, platform="<general>"):
        tokens = [platform, config.START_TOKEN]
        for word in caption.lower().split():
            tokens.append(word if word in self.word_to_idx else config.UNK_TOKEN)
        tokens.append(config.END_TOKEN)
        return [self.word_to_idx[t] for t in tokens]

    def decode(self, indices):
        words = []
        for idx in indices:
            word = self.idx_to_word.get(idx, config.UNK_TOKEN)
            if word == config.END_TOKEN:
                break
            if word not in [config.START_TOKEN, config.PAD_TOKEN]:
                words.append(word)
        return " ".join(words)


# ─── COCO Dataset ─────────────────────────────────────────────────────────────

class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_file, vocab, platform="<general>", transform=None):
        self.image_dir = image_dir
        self.vocab     = vocab
        self.platform  = platform
        self.transform = transform

        with open(annotation_file, "r") as f:
            data = json.load(f)

        self.id_to_filename = {
            img["id"]: img["file_name"] for img in data["images"]
        }
        self.samples = [
            (ann["image_id"], ann["caption"]) for ann in data["annotations"]
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, caption = self.samples[idx]
        image_path = os.path.join(self.image_dir, self.id_to_filename[image_id])
        image      = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        encoded        = self.vocab.encode(caption, self.platform)[:config.MAX_CAPTION_LEN]
        caption_tensor = torch.tensor(encoded, dtype=torch.long)
        return image, caption_tensor


# ─── Flickr8k Dataset ─────────────────────────────────────────────────────────

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_file, vocab, platform="<general>", transform=None):
        self.image_dir = image_dir
        self.vocab     = vocab
        self.platform  = platform
        self.transform = transform
        self.samples   = []

        with open(caption_file, "r") as f:
            next(f)  # skip header row (image,caption)
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # split on first comma only
                # captions can contain commas so we only split once
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue

                filename = parts[0].strip()
                caption  = parts[1].strip()
                self.samples.append((filename, caption))

        print(f"Loaded {len(self.samples)} image-caption pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, caption = self.samples[idx]
        image_path = os.path.join(self.image_dir, filename)
        image      = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        encoded        = self.vocab.encode(caption, self.platform)[:config.MAX_CAPTION_LEN]
        caption_tensor = torch.tensor(encoded, dtype=torch.long)
        return image, caption_tensor


# ─── Helpers ──────────────────────────────────────────────────────────────────

def collate_fn(batch):
    images, captions = zip(*batch)
    images  = torch.stack(images, dim=0)
    max_len = max(len(cap) for cap in captions)
    padded  = torch.zeros(len(captions), max_len, dtype=torch.long)

    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    return images, padded


def get_transform():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
    ])


def build_flickr_vocabulary(caption_file):
    print("Building Flickr8k vocabulary...")
    all_captions = []

    with open(caption_file, "r") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                all_captions.append(parts[1].strip())

    vocab = Vocabulary()
    vocab.build(all_captions)

    print(f"Vocabulary built! Total words: {len(vocab)}")
    return vocab


def build_vocabulary(annotation_file):
    print("Building vocabulary...")

    with open(annotation_file, "r") as f:
        data = json.load(f)

    all_captions = [ann["caption"] for ann in data["annotations"]]
    vocab        = Vocabulary()
    vocab.build(all_captions)

    print(f"Vocabulary built! Total words: {len(vocab)}")
    return vocab