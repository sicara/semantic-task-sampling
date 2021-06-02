from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from easyfsl.data_tools import EasySet
from easyfsl.data_tools.samplers import UniformTaskSampler
from easyfsl.methods import PrototypicalNetworks

train_set = EasySet(specs_file="./data/CUB/train.json", training=True)
train_sampler = UniformTaskSampler(
    train_set, n_way=5, n_shot=5, n_query=10, n_tasks=20
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

model.fit(train_loader, optimizer)

torch.save(model.state_dict(), Path("./data/models/trained_model.tar"))
