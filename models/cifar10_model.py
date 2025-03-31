import torch.nn as nn
import torchvision.models as models

def get_cifar10_model(pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model
