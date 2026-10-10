
import sys

import torch
from PIL import Image
from torchvision import transforms

from src.models.image_tagger import ImageTagger


# Temporary names based on dataset sample inspection
TAG_NAMES = [
    "sky", "clouds", "person", "water", "animal",
    "grass", "buildings", "window", "plants", "water/lake",
    "ocean", "road", "flowers", "sunset", "reflection",
    "rocks", "vehicle", "snow", "tree", "beach", "mountain"
]


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImageTagger(num_tags=21).to(device)
    model.load_state_dict(
        torch.load(
            "checkpoints/best_model.pth",
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    with Image.open(image_path) as img:
        image = transform(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.sigmoid(model(image))[0].cpu()

    print("\nPredicted Tags:")

    results = sorted(
        zip(TAG_NAMES, probabilities.tolist()),
        key=lambda item: item[1],
        reverse=True
    )

    for tag, score in results:
        print(f"{tag}: {score:.2%}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.predict <image_path>")
    else:
        predict(sys.argv[1])
