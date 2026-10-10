
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class NUSWIDEDataset(Dataset):

    def __init__(self, root, images_file, labels_file,
                 transform=None, indices=None):

        self.root = Path(root)
        self.transform = transform

        self.image_paths = (
            self.root / images_file
        ).read_text(encoding="utf-8").splitlines()

        self.labels = np.loadtxt(
            self.root / labels_file,
            dtype=np.float32
        )

        if len(self.image_paths) != len(self.labels):
            raise ValueError("Image and label counts do not match")

        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = self.labels[indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.root / self.image_paths[index]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.from_numpy(self.labels[index])

        return image, labels
