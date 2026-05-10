# SMILES-2026 Hallucination Detection Solution

## Reproducibility

The final run was executed on Google Colab with an NVIDIA A100 GPU. The solution is self-contained and uses the original `solution.py` entry point without modifying fixed infrastructure files.

To reproduce the submitted `results.json` and `predictions.csv`:

```bash
git clone https://github.com/Dmitrii-Topchii/Hallucination_detection.git
cd Hallucination_detection
pip install -r requirements.txt
python solution.py
```

The script loads `Qwen/Qwen2.5-0.5B`, extracts hidden states for `data/dataset.csv`, evaluates the probe with stratified folds, then extracts features for `data/test.csv` and writes `predictions.csv`.

Final validation summary from `results.json`:

| Metric | Value |
|---|---:|
| Majority baseline accuracy | 0.7010155506 |
| Probe test accuracy | 0.7227229451 |
| Probe test F1 | 0.8184676616 |
| Probe test AUROC | 0.7089793470 |
| Feature dimension | 20664 |
| Number of folds | 5 |

## Modified Components

Only the three allowed solution files were modified:

- `aggregation.py`
- `probe.py`
- `splitting.py`

The fixed files `solution.py`, `model.py`, and `evaluate.py` were not changed.

## Final Approach

The final solution uses Qwen hidden states as a source of internal evidence about the model response. Instead of relying on only the final hidden state at the last token, `aggregation.py` constructs a richer representation from several transformer layers and several token-pooling views.

The aggregation step combines:

- last-token hidden states from middle and late layers;
- mean-pooled hidden states over the full sequence;
- mean-pooled hidden states over the final 32 and final 8 real tokens;
- recency-weighted pooling over the final 16 real tokens;
- layer-difference vectors that capture representation drift;
- geometric/statistical features such as activation norms, cosine similarities, variance, and normalized sequence length.

This produced a feature vector of dimension `20664`.

The probe in `probe.py` is an ensemble of regularized scikit-learn classifiers rather than a single small neural network. The ensemble includes logistic regression, feature-selected logistic regression, ridge classifiers, PCA-compressed linear models, linear SVM-style models, LDA, and an elastic-net SGD classifier. The predicted hallucination probability is a weighted average across the successful estimators.

The decision threshold is tuned on validation folds during evaluation. For the final `predictions.csv` generation, where `solution.py` fits a final probe on the labeled data and then predicts the unlabeled competition set, the probe performs an internal out-of-fold threshold calibration using only the labeled training data.

The split strategy in `splitting.py` uses stratified 5-fold evaluation. This gives a more stable estimate than a single train/validation/test split while preserving the class balance in each fold.

## Why These Choices

The dataset is small, with 689 labeled samples, while hidden-state vectors are high-dimensional. This makes regularization and dimensionality reduction important. The final probe therefore avoids a large neural classifier and instead uses a set of simple, regularized estimators whose errors are averaged.

The response label depends on whether the answer is truthful or hallucinated, so the final tokens of the generated answer are likely to be especially informative. Tail pooling and recency-weighted pooling were added to preserve this signal while still keeping context from the full prompt-response sequence.

The most useful changes were:

- replacing single-layer last-token aggregation with multi-layer and tail-token aggregation;
- adding layer-difference and geometric features;
- using stratified 5-fold validation;
- replacing the simple MLP with a regularized ensemble;
- calibrating the classification threshold for accuracy.

## Experiments and Failed Attempts

The starting point was the repository baseline: final-layer, last-token aggregation with a small MLP probe. This was simple and fast, but it underused the hidden-state information and was sensitive to the class imbalance.

An intermediate version used a multi-layer feature vector of dimension `17976` and an ensemble probe. It achieved:

| Metric | Value |
|---|---:|
| Probe test accuracy | 0.7184068550 |
| Probe test F1 | 0.8140166531 |
| Probe test AUROC | 0.7090856752 |

The final version added recency-weighted answer-tail pooling and a stronger weighted ensemble with feature selection and out-of-fold threshold calibration. This improved the primary validation accuracy to `0.7227229451` and F1 to `0.8184676616`. AUROC stayed essentially unchanged, so the final version was selected because the competition primary metric is accuracy.

Some ideas were not kept as standalone final components:

- using only the final layer, because it lost useful middle-layer and representation-drift signal;
- relying only on high-dimensional raw concatenation, because the labeled dataset is too small for that to be stable;
- using only a neural MLP probe, because regularized linear/PCA models were more robust for this data size;
- optimizing only AUROC, because the task ranks by accuracy on `test.csv`.
