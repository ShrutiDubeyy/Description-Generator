# train.py
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.dataset import Flickr8kDataset, build_flickr_vocabulary, get_transform, collate_fn
from src.model import CaptionModel
import config


def setup():
    print("=" * 50)
    print("       IMAGE CAPTION GENERATOR")
    print("=" * 50)

    caption_file = os.path.join("data", "flickr8k", "captions.txt")
    image_dir    = os.path.join("data", "flickr8k", "Images")

    vocab = build_flickr_vocabulary(caption_file)
    print(f"\nVocabulary size: {len(vocab)} words")

    transform    = get_transform()
    full_dataset = Flickr8kDataset(
        image_dir    = image_dir,
        caption_file = caption_file,
        vocab        = vocab,
        platform     = "<general>",
        transform    = transform
    )

    print(f"Total samples: {len(full_dataset)}")

    train_size = int(0.9 * len(full_dataset))
    val_size   = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"Training samples:   {train_size}")
    print(f"Validation samples: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0
    )

    model  = CaptionModel(vocab_size=len(vocab))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # ── RESUME: load weights if checkpoint exists ──
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load(checkpoint_path)
        print(f"✅ Resumed from checkpoint: {checkpoint_path}")
    else:
        print("🆕 No checkpoint found, starting fresh.")

    print(f"\nUsing device: {device}")
    print(f"Model ready!")

    return model, train_loader, val_loader, vocab, device


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss    = 0
    total_batches = len(loader)

    for batch_idx, (images, captions) in enumerate(loader):

        images   = images.to(device)
        captions = captions.to(device)

        input_captions  = captions[:, :-1]
        target_captions = captions[:, 1:]

        predictions, alphas = model(images, input_captions)

        seq_len         = min(predictions.size(1), target_captions.size(1))
        predictions     = predictions[:, :seq_len, :]
        target_captions = target_captions[:, :seq_len]

        predictions     = predictions.reshape(-1, predictions.size(-1))
        target_captions = target_captions.reshape(-1)

        loss = criterion(predictions, target_captions)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            avg = total_loss / (batch_idx + 1)
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{total_batches} | Loss: {avg:.4f}")

    return total_loss / total_batches


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, captions in loader:

            images   = images.to(device)
            captions = captions.to(device)

            input_captions  = captions[:, :-1]
            target_captions = captions[:, 1:]

            predictions, alphas = model(images, input_captions)

            seq_len         = min(predictions.size(1), target_captions.size(1))
            predictions     = predictions[:, :seq_len, :]
            target_captions = target_captions[:, :seq_len]

            predictions     = predictions.reshape(-1, predictions.size(-1))
            target_captions = target_captions.reshape(-1)

            loss        = criterion(predictions, target_captions)
            total_loss += loss.item()

    return total_loss / len(loader)


def train():
    model, train_loader, val_loader, vocab, device = setup()

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.5
    )

    # ── RESUME: read metadata to know where to start ──
    best_val_loss    = float('inf')
    start_epoch      = 1
    checkpoint_path  = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    resume_meta_path = os.path.join(config.CHECKPOINT_DIR, "resume.txt")

    if os.path.exists(resume_meta_path):
        with open(resume_meta_path, "r") as f:
            lines         = f.readlines()
            start_epoch   = int(lines[0].strip()) + 1
            best_val_loss = float(lines[1].strip())
        print(f"▶️  Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    print("\nStarting training...\n")
    print("-" * 40)

    for epoch in range(start_epoch, config.EPOCHS + 1):

        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        val_loss = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Val Loss   : {val_loss:.4f}")
        print(f"  Time       : {epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

            model.save(checkpoint_path)

            # ── RESUME: save epoch + best loss to disk ──
            with open(resume_meta_path, "w") as f:
                f.write(f"{epoch}\n{best_val_loss}")

            print(f"  ✅ Best model saved! Val Loss: {best_val_loss:.4f}")

        print("-" * 40)

    print("\n🎉 Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved at: {config.CHECKPOINT_DIR}/best_model.pth")


if __name__ == "__main__":
    train()