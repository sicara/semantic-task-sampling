from torch import nn
from torchvision.models import resnet18

from easyfsl.resnet import resnet12

# Backbones configuration

TIERED_IMAGENET_NUM_TRAIN_CLASSES = 351


def get_pretrained_backbone_without_fc():
    backbone = resnet18(pretrained=True)
    backbone.fc = nn.Flatten()
    return backbone


BACKBONES_PER_DATASET = {
    "tiered_imagenet": resnet12(num_classes=TIERED_IMAGENET_NUM_TRAIN_CLASSES),
    "fungi": get_pretrained_backbone_without_fc(),
}

# Semantic Testbed Sampling

DEFAULT_ALPHA = 0.3830
DEFAULT_BETA_PENALTY = 100.0
