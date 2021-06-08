from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from easyfsl.data_tools import EasySet
from easyfsl.data_tools.samplers import SemanticTaskSampler
from easyfsl.methods import PrototypicalNetworks

train_set = EasySet(specs_file="./data/tiered_imagenet/train.json", training=True)
train_sampler = SemanticTaskSampler(
    train_set,
    n_way=5,
    n_shot=5,
    n_query=10,
    n_tasks=20,
    alpha=0.5,
    semantic_distances_csv=Path("data/tiered_imagenet/train_semantic_distances.csv"),
)
train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)

convolutional_network = resnet18(pretrained=False)
convolutional_network.fc = nn.Flatten()
model = PrototypicalNetworks(convolutional_network).cuda()

optimizer = Adam(params=model.parameters())

model.fit_multiple_epochs(train_loader, optimizer, n_epochs=2)

torch.save(model.state_dict(), Path("./data/models/trained_model.tar"))
