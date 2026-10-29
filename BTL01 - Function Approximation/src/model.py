"""Configurable MLP for the two-input, one-output function approximation task."""

from collections.abc import Sequence

from torch import Tensor, nn


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    """Fully connected MLP with configurable hidden-layer widths.

    Args:
        hidden: Neuron count for each hidden layer. For example, ``[16, 16]``
            creates two hidden layers with 16 neurons each. An empty sequence
            creates a linear model with no hidden layer.
        activation: Activation applied after each hidden linear layer.
    """

    def __init__(
        self,
        hidden: Sequence[int] = (16, 16),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if isinstance(hidden, (str, bytes)) or not isinstance(hidden, Sequence):
            raise TypeError("hidden must be a sequence of positive neuron counts")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in hidden
        ):
            raise ValueError("every hidden-layer size must be a positive integer")
        if activation not in _ACTIVATIONS:
            choices = ", ".join(sorted(_ACTIVATIONS))
            raise ValueError(f"unsupported activation {activation!r}; choose from {choices}")

        layers: list[nn.Module] = []
        input_size = 2
        activation_layer = _ACTIVATIONS[activation]
        for size in hidden:
            layers.extend((nn.Linear(input_size, size), activation_layer()))
            input_size = size
        layers.append(nn.Linear(input_size, 1))

        self.hidden = tuple(hidden)
        self.activation_name = activation
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Map a batch of ``(x1, x2)`` inputs shaped ``(n, 2)`` to ``(n, 1)``."""
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"input must have shape (n, 2), received {tuple(x.shape)}")
        return self.network(x)
