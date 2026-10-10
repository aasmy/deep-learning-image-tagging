
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


DATA_DIR = Path("data/raw/NUS-WIDE")

image_paths = (
    DATA_DIR / "database_img.txt"
).read_text(encoding="utf-8").splitlines()

labels = np.loadtxt(
    DATA_DIR / "database_label_onehot.txt",
    dtype=np.uint8
)

for tag in range(21):
    indices = np.where(labels[:, tag] == 1)[0]

    rng = np.random.default_rng(42 + tag)
    selected = rng.choice(
        indices,
        size=min(9, len(indices)),
        replace=False
    )

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    for ax, index in zip(axes.flat, selected):
        with Image.open(DATA_DIR / image_paths[index]) as img:
            ax.imshow(img.convert("RGB"))

        ax.axis("off")

    for ax in axes.flat[len(selected):]:
        ax.axis("off")

    fig.suptitle(f"Tag {tag} - Sample Images")
    plt.tight_layout()

    output_dir = Path("data/processed/tag_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_dir / f"tag_{tag}.png",
        dpi=120
    )
    plt.close(fig)

print("Saved 21 tag sample grids")
