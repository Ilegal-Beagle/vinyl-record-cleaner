from torch.distributions import Normal
import numpy as np
import torch.nn as nn
import torch
from torchinfo import summary

class Policy(nn.Module):
    def __init__(self, settings_shape, **kwargs):
        super().__init__(**kwargs)
        size = 32

        # pooling layer
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        
        # squashing down the input from the conv2d from 64 to settings size
        self.linear_layer = nn.Linear(64, settings_shape)

        # this starts wide, as the network is not confident in its configuration
        # after more runs are performed, it should become more confident and
        # lower the number. Parameter makes this change during training
        self.log_std = nn.Parameter(torch.full((settings_shape,), -0.5))

    # creates a normal distribution of settings to choose from
    def distribution(self, spec):
        features = self.body(spec)                                  # the features from the conv2d
        mean = torch.sigmoid(self.linear_layer(features))   # squash down the features
        return Normal(mean, self.log_std.exp())

    def sample(self, spec):
        dist = self.distribution(spec)
        action = dist.sample()                          # get a settings vector from distribution
        log_prob = dist.log_prob(action).sum(dim=-1)    # represents prob. as a log.
        return action.clamp(0, 1), log_prob             # adjust the values to a useable number [0, 1]

    @torch.no_grad()
    def act(self, spec):
        return self.distribution(spec).mean # get a non-random settings vector