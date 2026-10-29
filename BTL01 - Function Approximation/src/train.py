"""Reproducible PyTorch training with validation, early stopping, and logging."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import MLP


@dataclass
class TrainingResult:
    model: MLP
    history: list[dict[str, float]]
    best_epoch: int
    best_val_loss: float
    elapsed_seconds: float
    stopped_early: bool


def _as_xy(x: Any, y: Any, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert arrays to CPU float32 tensors and enforce the model's shapes."""
    x_tensor = torch.tensor(x, dtype=torch.float32, device="cpu")
    y_tensor = torch.tensor(y, dtype=torch.float32, device="cpu")
    if x_tensor.ndim != 2 or x_tensor.shape[1] != 2:
        raise ValueError(f"{name} features must have shape (n, 2)")
    if y_tensor.ndim == 1:
        y_tensor = y_tensor.unsqueeze(1)
    if y_tensor.ndim != 2 or y_tensor.shape[1] != 1:
        raise ValueError(f"{name} targets must have shape (n,) or (n, 1)")
    if len(x_tensor) == 0 or len(x_tensor) != len(y_tensor):
        raise ValueError(f"{name} features and targets must have the same non-zero length")
    if not torch.isfinite(x_tensor).all() or not torch.isfinite(y_tensor).all():
        raise ValueError(f"{name} features and targets must contain only finite values")
    return x_tensor, y_tensor


def _seed_everything(seed: int) -> torch.Generator:
    """Seed model initialization and batch shuffling for a repeatable run."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def train_model(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    *,
    hidden: tuple[int, ...] | list[int] = (16, 16),
    activation: str = "tanh",
    config: dict[str, Any] | None = None,
    seed: int = 0,
    log_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> TrainingResult:
    """Train an MLP and restore the weights with the lowest validation loss.

    ``config`` accepts the keys ``optimizer``, ``lr``, ``batch_size``,
    ``max_epochs`` and ``patience`` from the project's ``train`` config.
    Inputs have shapes ``(n, 2)`` and ``(n,)`` or ``(n, 1)``. Validation
    controls early stopping and checkpoint selection; test data is not passed
    here so it cannot influence training.
    """
    options = {
        "optimizer": "adam",
        "lr": 1e-3,
        "batch_size": 32,
        "max_epochs": 1000,
        "patience": 100,
    }
    if config is not None:
        options.update(config)

    optimizer_name = str(options["optimizer"]).lower()
    if optimizer_name not in {"adam", "sgd"}:
        raise ValueError("optimizer must be 'adam' or 'sgd'")
    learning_rate = float(options["lr"])
    batch_size = options["batch_size"]
    max_epochs = options["max_epochs"]
    patience = options["patience"]
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("lr must be a finite positive number")
    for name, value in (("batch_size", batch_size), ("max_epochs", max_epochs), ("patience", patience)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    train_x, train_y = _as_xy(X_train, y_train, "training")
    val_x, val_y = _as_xy(X_val, y_val, "validation")
    generator = _seed_everything(seed)
    target_device = torch.device(device)
    model = MLP(hidden=hidden, activation=activation).to(target_device)

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_x = val_x.to(target_device)
    val_y = val_y.to(target_device)
    loss_fn = nn.MSELoss()
    optimizer_class = torch.optim.Adam if optimizer_name == "adam" else torch.optim.SGD
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    stopped_early = False
    started_at = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(target_device)
            batch_y = batch_y.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            # Weight batch means by batch size so the epoch loss also handles
            # a final batch smaller than the configured batch size correctly.
            train_loss_sum += loss.item() * len(batch_x)
            train_count += len(batch_x)

        model.eval()
        # Validation measures generalization without tracking gradients or
        # modifying the parameters.
        with torch.no_grad():
            val_loss = loss_fn(model(val_x), val_y).item()
        train_loss = train_loss_sum / train_count
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            # Keep an independent CPU copy: state_dict tensors otherwise track
            # later optimizer updates and would not preserve the best epoch.
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stopped_early = epoch < max_epochs
                break

    elapsed_seconds = time.perf_counter() - started_at
    if best_state is None:
        raise RuntimeError("training completed without producing a finite validation checkpoint")
    # Early stopping can occur after the best epoch; return the best model,
    # not merely the weights from the final epoch.
    model.load_state_dict(best_state)
    model.to(target_device)
    model.eval()

    if log_path is not None:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).assign(epoch=lambda frame: frame["epoch"].astype(int)).to_csv(
            log_file, index=False
        )

    if checkpoint_path is not None:
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "hidden": list(model.hidden),
                "activation": model.activation_name,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "elapsed_seconds": elapsed_seconds,
                "seed": seed,
                "train_config": options,
            },
            checkpoint_file,
        )

    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        elapsed_seconds=elapsed_seconds,
        stopped_early=stopped_early,
    )


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MLP, dict[str, Any]]:
    """Load a checkpoint written by :func:`train_model`."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MLP(
        hidden=checkpoint["hidden"],
        activation=checkpoint["activation"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
