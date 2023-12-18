import torch
import torch.nn as nn


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_discriminator_out):
        return torch.mean((1 - gen_discriminator_out) ** 2)


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, gen):
        return torch.mean((1 - real) ** 2) + torch.mean(gen ** 2)
