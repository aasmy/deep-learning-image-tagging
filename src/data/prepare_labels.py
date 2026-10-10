
from pathlib import Path
import numpy as np

DATA_DIR = Path("data/raw/NUS-WIDE")


def inspect_split(name, images_file, labels_file):
    image_paths = (DATA_DIR / images_file).read_text(
        encoding="utf-8"
    ).splitlines()

    labels = np.loadtxt(
        DATA_DIR / labels_file,
        dtype=np.uint8
    )

    print(f"\n{name}")
    print("Images:", len(image_paths))
    print("Labels shape:", labels.shape)
    print("Values:", np.unique(labels))
    print("Tags per image (min):", labels.sum(axis=1).min())
    print("Tags per image (max):", labels.sum(axis=1).max())
    print("Tags per image (avg):", labels.sum(axis=1).mean())

    if len(image_paths) != len(labels):
        raise ValueError("Image and label counts do not match")

    missing = sum(
        not (DATA_DIR / path.strip()).is_file()
        for path in image_paths
    )

    print("Missing images:", missing)

    return labels.shape[1]


train_tags = inspect_split(
    "Database",
    "database_img.txt",
    "database_label_onehot.txt"
)

test_tags = inspect_split(
    "Test",
    "test_img.txt",
    "test_label_onehot.txt"
)

if train_tags != test_tags:
    raise ValueError("Train and test tag counts do not match")

print("\nTotal tag categories:", train_tags)
