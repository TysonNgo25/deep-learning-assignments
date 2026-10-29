"""Hàm Gaussian 2 chiều, sinh pool train 2000 / val 500 / test 100 với seed cố định, lưu CSV vào data/.

Phụ trách: Trúc (hạn 29/10).

Chạy:  python -m src.data
Dùng:  from src.data import load_train, load_split
       X_train, y_train = load_train(800)   # 800 dòng đầu của pool (thực nghiệm A)
       X_val, y_val = load_split("val")
"""
import numpy as np
import pandas as pd
import yaml

from src.paths import CONFIG_DIR, DATA_DIR

SPLITS = ("train_pool", "val", "test")
COLUMNS = ["x1", "x2", "y"]


def load_config(name="baseline.yaml"):
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def gaussian_2d(x, mu, cov):
    """f(x) = 1 / (2π √det C) · exp(-½ (x-μ)ᵀ C⁻¹ (x-μ)), tính cho từng dòng của x (shape (n, 2))."""
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)

    d = x - mu
    # Dạng toàn phương (x-μ)ᵀ C⁻¹ (x-μ) cho cả n điểm một lúc, kết quả shape (n,)
    quad = np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov), d)
    return np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))


def sample_split(n, rng, x1_range, x2_range, mu, cov):
    """Lấy n điểm phân bố đều trong miền chữ nhật, gắn nhãn y = f(x)."""
    x1 = rng.uniform(*x1_range, size=n)
    x2 = rng.uniform(*x2_range, size=n)
    y = gaussian_2d(np.column_stack([x1, x2]), mu, cov)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def generate_splits(cfg):
    """Sinh 3 tập độc lập. Mỗi tập có bộ sinh số ngẫu nhiên riêng tách ra từ cùng một seed,
    nên đổi kích thước một tập không làm thay đổi các tập còn lại."""
    sizes = {"train_pool": cfg["n_train_pool"], "val": cfg["n_val"], "test": cfg["n_test"]}
    rngs = np.random.default_rng(cfg["seed"]).spawn(len(SPLITS))
    return {
        name: sample_split(sizes[name], rng, cfg["x1_range"], cfg["x2_range"], cfg["mu"], cfg["cov"])
        for name, rng in zip(SPLITS, rngs)
    }


def save_splits(splits):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(DATA_DIR / f"{name}.csv", index=False)


def load_split(name):
    """Đọc data/<name>.csv, trả về X shape (n, 2) và y shape (n, 1), kiểu float32 để đưa thẳng vào torch."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    X = df[["x1", "x2"]].to_numpy(dtype=np.float32)
    y = df[["y"]].to_numpy(dtype=np.float32)
    return X, y


def load_train(n):
    """Lấy n dòng đầu của pool train. A = 200 nằm trọn trong A = 400, v.v., nên các cấu hình A chỉ khác nhau ở số mẫu."""
    X, y = load_split("train_pool")
    if not 0 < n <= len(X):
        raise ValueError(f"n_train = {n} ngoài khoảng 1..{len(X)}")
    return X[:n], y[:n]


def describe(splits, near_zero=0.01):
    """Bảng thống kê x1, x2, y cho từng tập (dùng cho mục 2 báo cáo)."""
    rows = []
    for name, df in splits.items():
        for col in COLUMNS:
            s = df[col]
            rows.append({"split": name, "column": col, "n": len(s), "mean": s.mean(), "std": s.std(),
                         "min": s.min(), "q25": s.quantile(0.25), "median": s.median(),
                         "q75": s.quantile(0.75), "max": s.max()})
        rows.append({"split": name, "column": f"ty_le_y<{near_zero}", "n": len(df),
                     "mean": (df["y"] < near_zero).mean()})
    return pd.DataFrame(rows)


def check(splits, cfg):
    """Kiểm tra nhanh: đúng kích thước, đúng miền, nhãn khớp công thức, không có điểm trùng giữa các tập."""
    peak = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cfg["cov"])))
    assert np.isclose(gaussian_2d([cfg["mu"]], cfg["mu"], cfg["cov"])[0], peak)

    for name, df in splits.items():
        assert len(df) == cfg[f"n_{name}"], name
        assert df["x1"].between(*cfg["x1_range"]).all(), name
        assert df["x2"].between(*cfg["x2_range"]).all(), name
        assert ((df["y"] > 0) & (df["y"] <= peak)).all(), name

    keys = [set(map(tuple, df[["x1", "x2"]].to_numpy())) for df in splits.values()]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not keys[i] & keys[j], f"Trùng điểm giữa {SPLITS[i]} và {SPLITS[j]}"
    return peak


def main():
    cfg = load_config()["data"]
    splits = generate_splits(cfg)
    peak = check(splits, cfg)
    save_splits(splits)

    stats = describe(splits)
    stats.to_csv(DATA_DIR / "stats.csv", index=False)

    print(f"det C = {np.linalg.det(cfg['cov']):.4f}, đỉnh f(μ) = {peak:.6f}")
    for name, df in splits.items():
        print(f"  {name:<10} {len(df):>5} mẫu -> {DATA_DIR / (name + '.csv')}")
    with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 120):
        print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
