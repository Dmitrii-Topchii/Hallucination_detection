from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    y_arr = np.asarray(y, dtype=int)
    idx = np.arange(len(y_arr))
    labels, counts = np.unique(y_arr, return_counts=True)
    if len(labels) < 2 or counts.min() < 5:
        idx_train_val, idx_test = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=y_arr if len(labels) > 1 else None,
        )
        relative_val = val_size / max(1e-6, 1.0 - test_size)
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=relative_val,
            random_state=random_state,
            stratify=y_arr[idx_train_val] if len(labels) > 1 else None,
        )
        return [(idx_train.astype(int), idx_val.astype(int), idx_test.astype(int))]
    n_splits = int(min(5, counts.min()))
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    splits = []
    relative_val = val_size / max(1e-6, 1.0 - (1.0 / n_splits))
    relative_val = float(np.clip(relative_val, 0.1, 0.3))
    for fold, (idx_train_val, idx_test) in enumerate(splitter.split(idx, y_arr)):
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=relative_val,
            random_state=random_state + fold,
            stratify=y_arr[idx_train_val],
        )
        splits.append((idx_train.astype(int), idx_val.astype(int), idx_test.astype(int)))
    return splits
