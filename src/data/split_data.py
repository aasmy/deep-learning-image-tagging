
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data/raw/NUS-WIDE")
OUTPUT_DIR = Path("data/splits")
RANDOM_SEED = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    database_paths = (
        (DATA_DIR / "database_img.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    test_paths = (
        (DATA_DIR / "test_img.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    labels = np.loadtxt(
        DATA_DIR / "database_label_onehot.txt",
        dtype=np.uint8
    )

    if len(database_paths) != len(labels):
        raise ValueError("Database images and labels do not match")

    if labels.ndim != 2 or labels.shape[1] != 21:
        raise ValueError("Expected 21 label columns")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary")

    if len(set(database_paths)) != len(database_paths):
        raise ValueError("Duplicate database image paths found")

    if set(database_paths) & set(test_paths):
        raise ValueError("Database and test contain overlapping paths")

    indices = np.arange(len(database_paths))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.10,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    np.save(OUTPUT_DIR / "train_indices.npy", train_indices)
    np.save(OUTPUT_DIR / "val_indices.npy", val_indices)

    print("Training images:", len(train_indices))
    print("Validation images:", len(val_indices))
    print("Test images:", len(test_paths))
    print("Number of labels:", labels.shape[1])

    train_counts = labels[train_indices].sum(axis=0)
    val_counts = labels[val_indices].sum(axis=0)

    print("\nLabel counts (train / validation):")
    for i in range(labels.shape[1]):
        print(f"Tag {i}: {train_counts[i]} / {val_counts[i]}")

    print("\nSplit files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
