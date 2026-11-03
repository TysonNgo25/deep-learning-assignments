"""Train the validation-selected model, save its checkpoint, and evaluate test once.

The existing test split was already evaluated for all C runs. The final metrics
are therefore reported transparently as a post-selection score on a previously
queried split, not as a fully untouched test-set estimate.
"""

import json

import torch

from src.data import load_config, load_split, load_train
from src.evaluate import evaluate_model
from src.paths import RESULTS_DIR
from src.train import load_checkpoint, train_model

OUT_DIR = RESULTS_DIR / "final"
CHECKPOINT_PATH = OUT_DIR / "selected_model.pt"
LOG_PATH = OUT_DIR / "logs" / "selected_model_seed1.csv"
METRICS_PATH = OUT_DIR / "final_metrics.json"


def main() -> None:
    if METRICS_PATH.exists():
        raise FileExistsError(
            f"Final test metrics already exist at {METRICS_PATH}; "
            "refusing to evaluate the test split again."
        )

    torch.set_num_threads(1)
    config = load_config()
    train_config = config["train"]
    selected_n_train = 2000
    selected_seed = 1
    options = {
        key: train_config[key]
        for key in ("optimizer", "lr", "batch_size", "max_epochs", "patience")
    }

    train_x, train_y = load_train(selected_n_train)
    val_x, val_y = load_split("val")
    training = train_model(
        train_x,
        train_y,
        val_x,
        val_y,
        hidden=train_config["hidden"],
        activation=train_config["activation"],
        config=options,
        seed=selected_seed,
        log_path=LOG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
    )

    model, checkpoint = load_checkpoint(CHECKPOINT_PATH)
    validation_metrics = evaluate_model(model, val_x, val_y)

    test_x, test_y = load_split("test")
    test_metrics = evaluate_model(model, test_x, test_y)

    result = {
        "selection": {
            "criterion": "validation MSE only; test metrics were not used for selection",
            "experiment": "A",
            "n_train": selected_n_train,
            "hidden": checkpoint["hidden"],
            "activation": checkpoint["activation"],
            "seed": selected_seed,
            "best_epoch": checkpoint["best_epoch"],
            "best_validation_loss": checkpoint["best_val_loss"],
            "validation_metrics": validation_metrics,
            "selected_config_mean_validation_mse": 1.681281041800533e-07,
            "selected_seed_validation_mse_from_A_log": 1.0453882026695283e-07,
        },
        "test_protocol": (
            "The existing test split had already been evaluated for all 12 "
            "experiment C runs. These are post-selection test metrics on that "
            "previously queried split, not an untouched independent test estimate."
        ),
        "test_metrics": test_metrics,
        "training": {
            "epochs_run": len(training.history),
            "elapsed_seconds": training.elapsed_seconds,
            "stopped_early": training.stopped_early,
            "checkpoint": str(CHECKPOINT_PATH.relative_to(OUT_DIR.parent)),
            "epoch_log": str(LOG_PATH.relative_to(OUT_DIR.parent)),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Selected A model: n_train={selected_n_train}, "
        f"hidden={checkpoint['hidden']}, seed={selected_seed}, "
        f"best_epoch={checkpoint['best_epoch']}"
    )
    print(f"Validation: {validation_metrics}")
    print(f"Test (previously queried split): {test_metrics}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
