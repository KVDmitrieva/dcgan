import numpy as np
from torch import nn
import torch.nn.functional as F


def init_weights(module):
    module_name = module.__class__.__name__
    if 'Conv' in module_name or "BatchNorm" in module_name:
        module.weight.data.normal_(mean=0.0, std=0.02)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        net = []
        for i in range(len(in_channels) - 1):
            net.append(nn.ConvTranspose2d(in_channels[i], out_channels[i], kernel_size[i], stride[i], padding[i]))
            net.append(nn.BatchNorm2d(out_channels[i]))
            net.append(nn.ReLU())

        net.append(nn.ConvTranspose2d(in_channels[-1], out_channels[-1], kernel_size[-1], stride[-1], padding[-1]))
        net.append(nn.Tanh())

        self.net = nn.Sequential(*net)
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        net = []
        for i in range(len(in_channels) - 1):
            net.append(nn.Conv2d(in_channels[i], out_channels[i], kernel_size[i], stride[i], padding[i]))
            net.append(nn.BatchNorm2d(out_channels[i]))
            net.append(nn.LeakyReLU(negative_slope=0.2))

        net.append(nn.Conv2d(in_channels[-1], out_channels[-1], kernel_size[-1], stride[-1], padding[-1]))

        self.net = nn.Sequential(*net)
        self.apply(init_weights)

    def forward(self, x):
        out = self.net(x).flatten(start_dim=1)
        return F.sigmoid(out)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
