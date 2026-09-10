"""Loader for the Persian Handwritten Digits dataset.

The dataset ships as a compact NumPy archive (PersianDigits.npz) with:
    images : uint8 array, shape (N, H, W)   -- grayscale, H = W = image_size (16)
    labels : int array,   shape (N,)        -- digit 0..9
    writers: str array,   shape (N,)        -- writer id, or "unknown"
    image_size : scalar int

Convention: white background, dark ink (pixel 255 = white, 0 = black ink).

Example
-------
    from load_persian_digits import load, stratified_split, writer_disjoint_split
    X, y, writers = load("PersianDigits.npz")          # X in [0,1], shape (N,1,H,W)
    (Xtr, ytr), (Xte, yte) = stratified_split(X, y, test_size=0.2, seed=0)
"""
import numpy as np


def load(npz_path="PersianDigits.npz", as_float=True, channel_dim=True):
    """Load the dataset.

    as_float=True  -> images scaled to [0,1] float32 (recommended for training)
    channel_dim=True -> shape (N,1,H,W); otherwise (N,H,W)
    Returns X, y, writers.
    """
    d = np.load(npz_path, allow_pickle=True)
    X = d["images"]
    y = d["labels"].astype(np.int64)
    writers = d["writers"].astype(str)
    if as_float:
        X = X.astype(np.float32) / 255.0
    if channel_dim:
        X = X[:, None, :, :]
    return X, y, writers


def stratified_split(X, y, test_size=0.2, seed=0):
    """Class-stratified train/test split (fixed by seed)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    return (X[tr], y[tr]), (X[te], y[te])


def writer_disjoint_split(X, y, writers, test_size=0.2, seed=0):
    """Writer-disjoint train/test split: no writer appears in both sets.

    Files whose writer is "unknown" are each treated as their own singleton group,
    so they are distributed but never cause leakage between known writers.
    """
    from sklearn.model_selection import GroupShuffleSplit
    groups = np.array([w if w != "unknown" else f"__unknown_{i}"
                       for i, w in enumerate(writers)], dtype=object)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    return (X[tr], y[tr]), (X[te], y[te])


if __name__ == "__main__":
    import collections
    X, y, w = load()
    print("images:", X.shape, "labels:", y.shape)
    print("per-class:", dict(sorted(collections.Counter(y.tolist()).items())))
    print("writers:", dict(sorted(collections.Counter(w.tolist()).items())))
