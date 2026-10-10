
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from src.data.dataset import NUSWIDEDataset
from src.models.image_tagger import ImageTagger


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = NUSWIDEDataset(
        root="data/raw/NUS-WIDE",
        images_file="test_img.txt",
        labels_file="test_label_onehot.txt",
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    model = ImageTagger(num_tags=21).to(device)

    model.load_state_dict(
        torch.load(
            "checkpoints/best_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.sigmoid(outputs)

            predictions.append(probabilities.cpu().numpy())
            actual.append(labels.numpy())

    y_true = np.vstack(actual)
    y_prob = np.vstack(predictions)
    y_pred = (y_prob >= 0.5).astype(int)

    precision = precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, average="micro", zero_division=0
    )

    map_score = average_precision_score(
        y_true, y_prob, average="macro"
    )

    print("\nEvaluation Results")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("mAP:", round(map_score, 4))


if __name__ == "__main__":
    main()
