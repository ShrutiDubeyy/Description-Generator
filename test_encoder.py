# # add at the bottom of test_encoder.py
# import torch
# from src.model import CaptionModel

# VOCAB_SIZE = 10009

# # create the complete model
# model = CaptionModel(vocab_size=VOCAB_SIZE)

# # fake batch of 2 images
# fake_images   = torch.randn(2, 3, 224, 224)

# # fake captions — 2 captions, each 15 words long
# fake_captions = torch.randint(0, VOCAB_SIZE, (2, 15))

# # forward pass — training mode
# predictions, alphas = model(fake_images, fake_captions)

# print("=== Complete Model Test ===")
# print("Predictions shape:", predictions.shape)
# print("Alphas shape:     ", alphas.shape)
# print("Expected preds:    torch.Size([2, 14, 10009])")
# print("Expected alphas:   torch.Size([2, 14, 49])")

from datasets import load_dataset

dataset = load_dataset("kkcosmos/instagram-images-with-captions", split="train")

print("Total samples:", len(dataset))
print("Columns:", dataset.column_names)

sample =dataset[0]
print(sample)