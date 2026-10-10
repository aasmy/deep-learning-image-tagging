
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.dataset import NUSWIDEDataset
from src.models.image_tagger import ImageTagger


DATA_DIR = "data/raw/NUS-WIDE"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.0001


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_indices = np.load("data/splits/train_indices.npy")
    val_indices = np.load("data/splits/val_indices.npy")

    train_dataset = NUSWIDEDataset(
        DATA_DIR,
        "database_img.txt",
        "database_label_onehot.txt",
        transform=transform,
        indices=train_indices
    )

    val_dataset = NUSWIDEDataset(
        DATA_DIR,
        "database_img.txt",
        "database_label_onehot.txt",
        transform=transform,
        indices=val_indices
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    model = ImageTagger(num_tags=21).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    Path("checkpoints").mkdir(exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_dataset)

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {train_loss:.4f}, "
            f"Validation Loss = {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pth"
            )
            print("Best model saved")


if __name__ == "__main__":
    main()
