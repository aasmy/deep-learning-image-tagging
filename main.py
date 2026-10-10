
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import NUSWIDEDataset


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = NUSWIDEDataset(
    root="data/raw/NUS-WIDE",
    images_file="database_img.txt",
    labels_file="database_label_onehot.txt",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(loader))

print("Images:", images.shape)
print("Labels:", labels.shape)
print("Labels example:", labels[0])
