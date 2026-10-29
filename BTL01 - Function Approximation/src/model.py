"""Configurable MLP for approximating a scalar-valued function."""

from collections.abc import Sequence

from torch import nn


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    """Fully connected network with two inputs and one linear output.

    ``hidden`` controls both the number of hidden layers and each layer's
    width; an empty sequence therefore creates a direct linear model.
    """

    def __init__(
        self,
        hidden: Sequence[int] = (16, 16),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if isinstance(hidden, (str, bytes)) or not isinstance(hidden, Sequence):
            raise TypeError("hidden must be a sequence of positive neuron counts")
        if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in hidden):
            raise ValueError("every hidden-layer size must be a positive integer")
        if activation not in _ACTIVATIONS:
            choices = ", ".join(sorted(_ACTIVATIONS))
            raise ValueError(f"unsupported activation {activation!r}; choose from {choices}")

        layers: list[nn.Module] = []
        input_size = 2
        activation_layer = _ACTIVATIONS[activation]
        for size in hidden:
            # Each hidden layer is affine + nonlinear; the output stays linear
            # so the network can predict any real-valued target.
            layers.extend((nn.Linear(input_size, size), activation_layer()))
            input_size = size
        layers.append(nn.Linear(input_size, 1))

        self.hidden = tuple(hidden)
        self.activation_name = activation
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Map a batch shaped ``(n, 2)`` to predictions shaped ``(n, 1)``."""
        return self.network(x)
