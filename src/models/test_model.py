
import torch

from src.models.image_tagger import ImageTagger


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = ImageTagger(num_tags=21).to(device)
    model.eval()

    images = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad():
        logits = model(images)
        probabilities = torch.sigmoid(logits)

    print("Input shape:", images.shape)
    print("Output shape:", logits.shape)
    print("Probabilities shape:", probabilities.shape)
    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    ))


if __name__ == "__main__":
    main()
