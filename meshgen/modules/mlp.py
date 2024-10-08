import torch
import torch.nn as nn
import itertools


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class MlpDecoder(nn.Module):
    """
    Triplane decoder that gives RGB and sigma values from sampled features.
    Using ReLU here instead of Softplus in the original implementation.

    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            nn.SiLU(),
            *itertools.chain(
                *[
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                    ]
                    for _ in range(num_layers - 2)
                ]
            ),
            nn.Linear(hidden_dim, 1),
        )
        # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):

        return self.net(x).squeeze(-1)
