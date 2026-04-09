# src/platform_dataset.py

import os
import csv
from PIL import Image

import torch
from torch.utils.data import Dataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class PlatformCaptionDataset(Dataset):
    def __init__(self, image_dir, csv_file, vocab, transform=None):
        """
        image_dir : folder with all flickr images
        csv_file  : our generated training_data.csv
        vocab     : vocabulary object
        transform : image preprocessing
        """
        self.image_dir = image_dir
        self.vocab     = vocab
        self.transform = transform
        self.samples   = []

        print("Loading platform caption dataset...")

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "filename": row["filename"],
                    "platform": row["platform"],
                    "caption" : row["caption"]
                })

        print(f"Loaded {len(self.samples)} platform caption pairs!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        filename = sample["filename"]
        platform = sample["platform"]
        caption  = sample["caption"]

        # ── Load image ────────────────────────────────────────────────────────
        image_path = os.path.join(self.image_dir, filename)

        # handle missing images gracefully
        if not os.path.exists(image_path):
            # return a black image if file missing
            image = Image.new("RGB", (224, 224), color=0)
        else:
            image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # ── Encode caption ────────────────────────────────────────────────────
        # add platform token + start + words + end
        platform_token = f"<{platform}>"
        encoded        = self.vocab.encode(caption, platform_token)
        encoded        = encoded[:config.MAX_CAPTION_LEN]
        caption_tensor = torch.tensor(encoded, dtype=torch.long)

        return image, caption_tensor