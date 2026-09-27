"""Reproducible Ames house price experiments with scikit-learn and PyTorch."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
FEATURE_SETS = ("raw", "area", "area_age_bath", "all")
MLP_FEATURE_SETS = ("raw", "all")
ROOT = Path(__file__).resolve().parent


def make_features(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Build house attributes available before sale, without using SalePrice."""
    if variant not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {variant}")
    x = frame.drop(columns=["SalePrice", "Id"], errors="ignore").copy()
    x["MSSubClass"] = x["MSSubClass"].astype("string")
    if variant == "raw":
        return x

    def number(name: str) -> pd.Series:
        return pd.to_numeric(x[name], errors="coerce").fillna(0)

    x["TotalHouseSF"] = number("GrLivArea") + number("TotalBsmtSF")
    x["TotalPorchSF"] = sum(
        (number(c) for c in ("OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch")),
        start=pd.Series(0, index=x.index),
    )
    if variant == "area":
        return x

    x["TotalBath"] = (
        number("FullBath") + 0.5 * number("HalfBath")
        + number("BsmtFullBath") + 0.5 * number("BsmtHalfBath")
    )
    x["HouseAgeAtSale"] = (number("YrSold") - number("YearBuilt")).clip(lower=0)
    x["YearsSinceRemodel"] = (number("YrSold") - number("YearRemodAdd")).clip(lower=0)
    if variant == "area_age_bath":
        return x

    x["QualityLivingArea"] = number("OverallQual") * number("GrLivArea")
    x["QualityTotalArea"] = number("OverallQual") * x["TotalHouseSF"]
    return x


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("number", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("category", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Missing",
                                         keep_empty_features=True)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        sparse_threshold=0,
    )


def metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    # Kaggle scores RMSE between natural logarithms of positive sale prices.
    return {
        "log_rmse": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_usd": float(mean_absolute_error(np.exp(y_true_log), np.exp(y_pred_log))),
    }


def fit_ridge(x_train: pd.DataFrame, y_train_log: np.ndarray) -> Pipeline:
    model = Pipeline([
        ("prepare", make_preprocessor(x_train)),
        ("regressor", Ridge(alpha=20.0)),
    ])
    model.fit(x_train, y_train_log)
    return model


def fit_gradient_boosting(x_train: pd.DataFrame, y_train_log: np.ndarray) -> Pipeline:
    model = Pipeline([
        ("prepare", make_preprocessor(x_train)),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=15, l2_regularization=1, random_state=SEED
        )),
    ])
    model.fit(x_train, y_train_log)
    return model


class MLP(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


class FittedMLP:
    def __init__(self, preprocessor: ColumnTransformer, network: MLP,
                 target_mean: float, target_std: float, epochs: int):
        self.preprocessor = preprocessor
        self.network = network.eval()
        self.target_mean = target_mean
        self.target_std = target_std
        self.epochs = epochs

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(x).astype(np.float32)
        with torch.no_grad():
            values = self.network(torch.from_numpy(transformed)).numpy()
        return values * self.target_std + self.target_mean


def fit_mlp(x_train: pd.DataFrame, y_train_log: np.ndarray, seed: int = SEED,
            max_epochs: int = 250) -> FittedMLP:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    inner_train, inner_valid = train_test_split(
        np.arange(len(x_train)), test_size=0.15, random_state=seed
    )
    # The validation fold is excluded from every fitted imputer, encoder, and scaler.
    preprocessor = make_preprocessor(x_train.iloc[inner_train])
    a = preprocessor.fit_transform(x_train.iloc[inner_train]).astype(np.float32)
    b = preprocessor.transform(x_train.iloc[inner_valid]).astype(np.float32)
    target_mean = float(y_train_log[inner_train].mean())
    target_std = float(y_train_log[inner_train].std())
    train_target = ((y_train_log[inner_train] - target_mean) / target_std).astype(np.float32)
    valid_target = ((y_train_log[inner_valid] - target_mean) / target_std).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(a), torch.from_numpy(train_target)),
        batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    network = MLP(a.shape[1])
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.001)
    loss_fn = nn.MSELoss()
    valid_x, valid_y = torch.from_numpy(b), torch.from_numpy(valid_target)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        network.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(network(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            valid_loss = loss_fn(network(valid_x), valid_y).item()
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
            best_epoch, stale = epoch, 0
        else:
            stale += 1
            if stale >= 25:
                break
    assert best_state is not None
    network.load_state_dict(best_state)
    return FittedMLP(preprocessor, network, target_mean, target_std, best_epoch)


def evaluate_cv(x: pd.DataFrame, y_log: np.ndarray) -> pd.DataFrame:
    rows = []
    folds = KFold(n_splits=3, shuffle=True, random_state=SEED)
    for model_name, fit_model in (("Ridge", fit_ridge),
                                  ("HistGradientBoosting", fit_gradient_boosting)):
        for feature_set in FEATURE_SETS:
            features = make_features(x, feature_set)
            for fold, (training, validation) in enumerate(folds.split(features), start=1):
                fitted = fit_model(features.iloc[training], y_log[training])
                scores = metrics(y_log[validation], fitted.predict(features.iloc[validation]))
                rows.append({"model": model_name, "features": feature_set,
                             "fold": fold, **scores})
                print(f"CV {model_name} {feature_set} fold {fold}: "
                      f"{scores['log_rmse']:.4f}", flush=True)
    for feature_set in MLP_FEATURE_SETS:
        features = make_features(x, feature_set)
        for fold, (training, validation) in enumerate(folds.split(features), start=1):
            fitted = fit_mlp(features.iloc[training], y_log[training], seed=SEED + fold)
            scores = metrics(y_log[validation], fitted.predict(features.iloc[validation]))
            rows.append({"model": "PyTorch MLP", "features": feature_set,
                         "fold": fold, "epochs": fitted.epochs, **scores})
            print(f"CV MLP {feature_set} fold {fold}: {scores['log_rmse']:.4f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data / "train.csv")
    competition_test = pd.read_csv(args.data / "test.csv")
    if train.shape != (1460, 81) or competition_test.shape != (1459, 80):
        raise ValueError("Unexpected Kaggle House Prices dataset dimensions")
    if set(train.columns) - {"SalePrice"} != set(competition_test.columns):
        raise ValueError("Train and test columns differ")
    y_log = np.log(train["SalePrice"].to_numpy(dtype=float))
    development, holdout = train_test_split(
        np.arange(len(train)), test_size=0.2, random_state=SEED
    )
    cv = evaluate_cv(train.iloc[development], y_log[development])
    cv.to_csv(args.output / "cv_results.csv", index=False)
    summary = (cv.groupby(["model", "features"], as_index=False)
               .agg(log_rmse_mean=("log_rmse", "mean"),
                    log_rmse_std=("log_rmse", "std"), mae_usd_mean=("mae_usd", "mean"))
               .sort_values("log_rmse_mean"))
    summary.to_csv(args.output / "cv_summary.csv", index=False)
    holdout_rows, predictions = [], pd.DataFrame({
        "Id": train.iloc[holdout]["Id"].to_numpy(),
        "SalePrice": train.iloc[holdout]["SalePrice"].to_numpy(),
    })
    fitters = {"Ridge": fit_ridge, "HistGradientBoosting": fit_gradient_boosting,
               "PyTorch MLP": fit_mlp}
    for model_name, fit_model in fitters.items():
        choice = summary[summary["model"] == model_name].iloc[0]
        feature_set = choice["features"]
        x_dev = make_features(train.iloc[development], feature_set)
        x_holdout = make_features(train.iloc[holdout], feature_set)
        fitted = fit_model(x_dev, y_log[development])
        predicted_log = fitted.predict(x_holdout)
        scores = metrics(y_log[holdout], predicted_log)
        holdout_rows.append({"model": model_name, "features": feature_set, **scores})
        predictions[model_name.replace(" ", "_") + "_prediction"] = np.exp(predicted_log)
        print(f"HOLDOUT {model_name} {feature_set}: {scores['log_rmse']:.4f}", flush=True)
    pd.DataFrame(holdout_rows).to_csv(args.output / "holdout_results.csv", index=False)
    predictions.to_csv(args.output / "holdout_predictions.csv", index=False)

    winner = summary.iloc[0]
    winner_features = winner["features"]
    full_features = make_features(train, winner_features)
    unlabeled_features = make_features(competition_test, winner_features)
    fitted = fitters[winner["model"]](full_features, y_log)
    submission = pd.DataFrame({
        "Id": competition_test["Id"].to_numpy(),
        "SalePrice": np.exp(fitted.predict(unlabeled_features)),
    })
    submission.to_csv(args.output / "submission.csv", index=False)
    info = {
        "seed": SEED, "train_rows": len(train), "competition_test_rows": len(competition_test),
        "development_rows": len(development), "holdout_rows": len(holdout),
        "numeric_columns": len(make_features(train, "raw").select_dtypes(include="number").columns),
        "categorical_columns": len(make_features(train, "raw").select_dtypes(exclude="number").columns),
        "winner_by_development_cv": {"model": winner["model"], "features": winner_features,
                                     "log_rmse_mean": float(winner["log_rmse_mean"])},
        "holdout_not_used_in_cv": True,
        "competition_test_has_no_labels": True,
    }
    (args.output / "run_summary.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print("Saved results to", args.output, flush=True)


if __name__ == "__main__":
    main()
