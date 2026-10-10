
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class ImageTagger(nn.Module):

    def __init__(self, num_tags=21):
        super().__init__()

        self.model = vgg19(weights=VGG19_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_tags)

    def forward(self, images):
        return self.model(images)
