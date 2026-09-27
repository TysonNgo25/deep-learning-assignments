"""Focused checks for feature construction, leakage, and exported predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

from run_experiments import make_features, make_preprocessor


ROOT = Path(__file__).resolve().parent


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    source = train.iloc[:1]
    raw = make_features(source, "raw")
    engineered = make_features(source, "all")
    assert "Id" not in raw and "SalePrice" not in raw
    assert engineered["TotalHouseSF"].iloc[0] == (
        source["GrLivArea"].iloc[0] + source["TotalBsmtSF"].iloc[0]
    )
    assert engineered["HouseAgeAtSale"].iloc[0] == max(
        0, source["YrSold"].iloc[0] - source["YearBuilt"].iloc[0]
    )

    # Fitted statistics come only from the training rows, even when the held-out
    # rows contain a very large numerical value and a new categorical level.
    first = make_features(train.iloc[:100], "raw")
    unseen = first.iloc[:1].copy()
    unseen["LotFrontage"] = 1_000_000
    unseen["MSZoning"] = "UNSEEN_CATEGORY"
    preprocessor = make_preprocessor(first)
    fitted = preprocessor.fit_transform(first)
    transformed = preprocessor.transform(unseen)
    assert fitted.shape[1] == transformed.shape[1]
    numeric_names = first.select_dtypes(include="number").columns.tolist()
    position = numeric_names.index("LotFrontage")
    observed_median = preprocessor.named_transformers_["number"].named_steps[
        "impute"
    ].statistics_[position]
    assert np.isclose(observed_median, first["LotFrontage"].median())

    results = ROOT / "results"
    cv = pd.read_csv(results / "cv_results.csv")
    holdout = pd.read_csv(results / "holdout_results.csv")
    submission = pd.read_csv(results / "submission.csv")
    assert len(cv) == 30 and set(cv["fold"]) == {1, 2, 3}
    assert set(holdout["model"]) == {"Ridge", "HistGradientBoosting", "PyTorch MLP"}
    assert submission.columns.tolist() == ["Id", "SalePrice"]
    assert len(submission) == len(test) == 1459
    assert submission["Id"].equals(test["Id"])
    assert submission["SalePrice"].notna().all()
    assert np.isfinite(submission["SalePrice"]).all()
    assert (submission["SalePrice"] > 0).all()
    print("All feature, leakage, and output checks passed.")


if __name__ == "__main__":
    main()
