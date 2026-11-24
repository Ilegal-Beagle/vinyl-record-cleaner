from torch.distributions import Normal
import torch.nn as nn
import torch

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
        self.log_std = nn.Parameter(torch.full((settings_shape,), -1.5))

    # creates a normal distribution of settings to choose from
    def distribution(self, spec):
        spec = torch.as_tensor(spec, dtype=torch.float32)
        if spec.ndim == 2:            # (freq, time) -> (1, 1, freq, time)
            spec = spec[None, None]
        elif spec.ndim == 3:          # (B, freq, time) -> (B, 1, freq, time)
            spec = spec[:, None]
        spec = torch.log1p(spec)      # compress magnitude range (optional, but helps)
        features = self.body(spec)
        mean = torch.sigmoid(self.linear_layer(features))
        return Normal(mean, self.log_std.exp())

    def sample(self, spec):
        dist = self.distribution(spec)
        action = dist.sample()                          # get a settings vector from distribution
        log_prob = dist.log_prob(action).sum(dim=-1)    # represents prob. as a log.
        return action.clamp(0, 1), log_prob             # adjust the values to a useable number [0, 1]

    @torch.no_grad()
    def act(self, spec):
        return self.distribution(spec).mean # get a non-random settings vector