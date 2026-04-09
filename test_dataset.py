# test_dataset.py
from datasets import load_dataset

print("Loading real Instagram dataset...")

dataset = load_dataset(
    "kkcosmos/instagram-images-with-captions",
    split="train"
)

print(f"Total samples: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
print("\nSample captions:")
print("-" * 50)

for i in range(5):
    sample = dataset[i]
    print(f"\nCaption: {sample['caption']}")