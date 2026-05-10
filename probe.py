from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._models: list = []
        self._threshold: float = 0.5
        self._prior: float = 0.5

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X_arr, nan=0.0, posinf=1e6, neginf=-1e6)

    def _candidate_models(self, n_samples: int, n_features: int) -> list:
        pca_main = max(2, min(96, n_samples - 2, n_features))
        pca_small = max(2, min(48, n_samples - 2, n_features))
        return [
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
            make_pipeline(
                StandardScaler(),
                PCA(
                    n_components=pca_main,
                    random_state=19,
                    svd_solver="randomized",
                    whiten=True,
                ),
                LogisticRegression(
                    C=0.35,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=23,
                    solver="liblinear",
                ),
            ),
            make_pipeline(
                StandardScaler(),
                PCA(
                    n_components=pca_small,
                    random_state=29,
                    svd_solver="randomized",
                    whiten=False,
                ),
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
            make_pipeline(
                StandardScaler(),
                PCA(
                    n_components=pca_small,
                    random_state=31,
                    svd_solver="randomized",
                    whiten=True,
                ),
                SVC(
                    C=1.25,
                    class_weight="balanced",
                    gamma="scale",
                    probability=True,
                    random_state=37,
                ),
            ),
        ]

    def _set_prior_threshold(self, probs: np.ndarray) -> None:
        if probs.size == 0:
            self._threshold = 0.5
            return
        target = float(np.clip(self._prior, 0.05, 0.95))
        self._threshold = float(np.quantile(probs, 1.0 - target))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("HallucinationProbe uses sklearn estimators for inference.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_arr = self._prepare_X(X)
        y_arr = np.asarray(y, dtype=int)
        self._prior = float(np.mean(y_arr)) if y_arr.size else 0.5
        self._models = []
        for model in self._candidate_models(X_arr.shape[0], X_arr.shape[1]):
            try:
                model.fit(X_arr, y_arr)
                self._models.append(model)
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
        self._set_prior_threshold(self.predict_proba(X_arr)[:, 1])
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        y_arr = np.asarray(y_val, dtype=int)
        probs = self.predict_proba(X_val)[:, 1]
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
        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("Call fit() before predict_proba().")
        X_arr = self._prepare_X(X)
        parts = []
        for model in self._models:
            if hasattr(model, "predict_proba"):
                parts.append(model.predict_proba(X_arr)[:, 1])
            else:
                scores = model.decision_function(X_arr)
                parts.append(1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0))))
        prob_pos = np.clip(np.mean(np.vstack(parts), axis=0), 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - prob_pos, prob_pos])
