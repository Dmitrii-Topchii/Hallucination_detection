from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC, SVC


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._models: list = []
        self._weights: list[float] = []
        self._threshold: float = 0.5
        self._prior: float = 0.5
        self._fit_X: np.ndarray | None = None
        self._fit_y: np.ndarray | None = None
        self._threshold_tuned: bool = False

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X_arr, nan=0.0, posinf=1e6, neginf=-1e6)

    def _candidate_models(self, n_samples: int, n_features: int) -> list:
        pca_main = max(2, min(96, n_samples - 2, n_features))
        pca_mid = max(2, min(64, n_samples - 2, n_features))
        pca_small = max(2, min(40, n_samples - 2, n_features))
        select_large = min(4096, n_features)
        select_mid = min(2048, n_features)
        return [
            (
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=0.025,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=11,
                        solver="liblinear",
                    ),
                ),
                1.0,
            ),
            (
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=0.075,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=13,
                        solver="liblinear",
                    ),
                ),
                1.25,
            ),
            (
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=0.15,
                        class_weight=None,
                        max_iter=5000,
                        random_state=17,
                        solver="liblinear",
                    ),
                ),
                0.75,
            ),
            (
                make_pipeline(
                    SelectKBest(f_classif, k=select_large),
                    StandardScaler(),
                    LogisticRegression(
                        C=0.08,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=19,
                        solver="liblinear",
                    ),
                ),
                1.35,
            ),
            (
                make_pipeline(
                    SelectKBest(f_classif, k=select_mid),
                    StandardScaler(),
                    LogisticRegression(
                        C=0.22,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=23,
                        solver="liblinear",
                    ),
                ),
                1.1,
            ),
            (
                make_pipeline(
                    StandardScaler(),
                    RidgeClassifier(alpha=25.0, class_weight="balanced"),
                ),
                0.9,
            ),
            (
                make_pipeline(
                    StandardScaler(),
                    RidgeClassifier(alpha=90.0, class_weight=None),
                ),
                0.65,
            ),
            (
                make_pipeline(
                    StandardScaler(),
                    PCA(
                        n_components=pca_main,
                        random_state=29,
                        svd_solver="randomized",
                        whiten=True,
                    ),
                    LogisticRegression(
                        C=0.35,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=31,
                        solver="liblinear",
                    ),
                ),
                1.0,
            ),
            (
                make_pipeline(
                    StandardScaler(),
                    PCA(
                        n_components=pca_mid,
                        random_state=37,
                        svd_solver="randomized",
                        whiten=True,
                    ),
                    LinearSVC(
                        C=0.08,
                        class_weight="balanced",
                        dual=True,
                        max_iter=6000,
                        random_state=41,
                    ),
                ),
                0.85,
            ),
            (
                make_pipeline(
                    StandardScaler(),
                    PCA(
                        n_components=pca_small,
                        random_state=43,
                        svd_solver="randomized",
                        whiten=False,
                    ),
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                ),
                0.75,
            ),
            (
                make_pipeline(
                    StandardScaler(),
                    PCA(
                        n_components=pca_small,
                        random_state=47,
                        svd_solver="randomized",
                        whiten=True,
                    ),
                    SVC(
                        C=1.1,
                        class_weight="balanced",
                        gamma="scale",
                        probability=False,
                        random_state=53,
                    ),
                ),
                0.65,
            ),
            (
                make_pipeline(
                    StandardScaler(),
                    SGDClassifier(
                        alpha=0.00015,
                        class_weight="balanced",
                        early_stopping=True,
                        l1_ratio=0.15,
                        loss="log_loss",
                        max_iter=3000,
                        n_iter_no_change=20,
                        penalty="elasticnet",
                        random_state=59,
                    ),
                ),
                0.55,
            ),
        ]

    def _threshold_models(self, n_samples: int, n_features: int) -> list:
        pca_small = max(2, min(48, n_samples - 2, n_features))
        select_mid = min(2048, n_features)
        return [
            (
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=0.075,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=61,
                        solver="liblinear",
                    ),
                ),
                1.0,
            ),
            (
                make_pipeline(
                    SelectKBest(f_classif, k=select_mid),
                    StandardScaler(),
                    LogisticRegression(
                        C=0.16,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=67,
                        solver="liblinear",
                    ),
                ),
                1.15,
            ),
            (
                make_pipeline(
                    StandardScaler(),
                    RidgeClassifier(alpha=35.0, class_weight="balanced"),
                ),
                0.8,
            ),
            (
                make_pipeline(
                    StandardScaler(),
                    PCA(
                        n_components=pca_small,
                        random_state=71,
                        svd_solver="randomized",
                        whiten=True,
                    ),
                    LogisticRegression(
                        C=0.4,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=73,
                        solver="liblinear",
                    ),
                ),
                0.9,
            ),
        ]

    def _set_prior_threshold(self, probs: np.ndarray) -> None:
        if probs.size == 0:
            self._threshold = 0.5
            return
        target = float(np.clip(self._prior, 0.05, 0.95))
        self._threshold = float(np.quantile(probs, 1.0 - target))

    def _estimator_score(self, model, X_arr: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_arr)[:, 1]
        scores = model.decision_function(X_arr)
        scores = np.asarray(scores)
        if scores.ndim == 2:
            scores = scores[:, -1]
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))

    def _weighted_average(self, models: list, weights: list[float], X_arr: np.ndarray) -> np.ndarray:
        parts = []
        valid_weights = []
        for model, weight in zip(models, weights):
            parts.append(self._estimator_score(model, X_arr))
            valid_weights.append(weight)
        scores = np.vstack(parts)
        weight_arr = np.asarray(valid_weights, dtype=np.float64)
        return np.average(scores, axis=0, weights=weight_arr)

    def _best_threshold(self, probs: np.ndarray, y_arr: np.ndarray) -> float:
        grid = np.linspace(0.01, 0.99, 199)
        if probs.size:
            quantiles = np.quantile(probs, np.linspace(0.01, 0.99, 99))
            candidates = np.unique(np.concatenate([grid, quantiles, probs]))
        else:
            candidates = grid
        best_threshold = self._threshold
        best_accuracy = -1.0
        best_f1 = -1.0
        for threshold in candidates:
            pred = (probs >= threshold).astype(int)
            acc = accuracy_score(y_arr, pred)
            f1 = f1_score(y_arr, pred, zero_division=0)
            if acc > best_accuracy or (acc == best_accuracy and f1 > best_f1):
                best_accuracy = acc
                best_f1 = f1
                best_threshold = float(threshold)
        return best_threshold

    def _calibrate_threshold_from_fit(self) -> None:
        if self._threshold_tuned or self._fit_X is None or self._fit_y is None:
            return
        X_arr = self._fit_X
        y_arr = self._fit_y
        labels, counts = np.unique(y_arr, return_counts=True)
        if len(labels) < 2 or counts.min() < 5:
            self._set_prior_threshold(self.predict_proba(X_arr)[:, 1])
            self._threshold_tuned = True
            return
        n_splits = int(min(5, counts.min()))
        oof = np.zeros(len(y_arr), dtype=np.float64)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=83,
        )
        for train_idx, val_idx in splitter.split(X_arr, y_arr):
            fold_models = []
            fold_weights = []
            for model, weight in self._threshold_models(len(train_idx), X_arr.shape[1]):
                try:
                    model.fit(X_arr[train_idx], y_arr[train_idx])
                    fold_models.append(model)
                    fold_weights.append(weight)
                except Exception:
                    continue
            if fold_models:
                oof[val_idx] = self._weighted_average(
                    fold_models,
                    fold_weights,
                    X_arr[val_idx],
                )
            else:
                oof[val_idx] = self._prior
        self._threshold = self._best_threshold(oof, y_arr)
        self._threshold_tuned = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("HallucinationProbe uses sklearn estimators for inference.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_arr = self._prepare_X(X)
        y_arr = np.asarray(y, dtype=int)
        self._prior = float(np.mean(y_arr)) if y_arr.size else 0.5
        self._models = []
        self._weights = []
        self._fit_X = X_arr.copy()
        self._fit_y = y_arr.copy()
        self._threshold_tuned = False
        for model, weight in self._candidate_models(X_arr.shape[0], X_arr.shape[1]):
            try:
                model.fit(X_arr, y_arr)
                self._models.append(model)
                self._weights.append(weight)
            except Exception:
                continue
        if not self._models:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=41,
                    solver="liblinear",
                ),
            )
            model.fit(X_arr, y_arr)
            self._models.append(model)
            self._weights.append(1.0)
        self._set_prior_threshold(self.predict_proba(X_arr)[:, 1])
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        y_arr = np.asarray(y_val, dtype=int)
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = self._best_threshold(probs, y_arr)
        self._threshold_tuned = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        self._calibrate_threshold_from_fit()
        return (probs >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("Call fit() before predict_proba().")
        X_arr = self._prepare_X(X)
        prob_pos = np.clip(
            self._weighted_average(self._models, self._weights, X_arr),
            1e-6,
            1.0 - 1e-6,
        )
        return np.column_stack([1.0 - prob_pos, prob_pos])
