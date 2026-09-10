# Persian Handwritten Digits — Dataset & CNN Model

A dataset of **2,560 handwritten Persian (Farsi) digit images** collected from multiple
writers with diverse handwriting styles, together with a trained convolutional neural
network and a fully reproducible, leak-free evaluation pipeline.

This release accompanies the manuscript *"Persian Handwritten Digit Recognition: A
Comparative Study of Deep Learning and Machine Learning Models"* (under review).

---

## What's here

| File / folder | Description |
|---|---|
| `dataset/0` … `dataset/9` | Raw images, one folder per digit (16×16 grayscale JPG). |
| `PersianDigits.npz` | Compact archive: `images` (uint8, N×16×16), `labels` (0–9), `writers`. |
| `PersianDigits.csv` | One row per image: `label`, `writer`, then 256 pixel values (Table 1 style). |
| `model/persian_digit_cnn.pt` | Trained CNN weights (PyTorch `state_dict`). |
| `model/preprocessing.json` | Input size, normalization constants, architecture, reported accuracy. |
| `model.py` | The CNN architecture (shared by training and inference). |
| `load_persian_digits.py` | Loader with stratified and **writer-disjoint** split helpers. |
| `predict.py` | Run the trained model on your own image(s). |
| `persian_digits_rigorous_cnn.ipynb` | Full training / evaluation notebook (CV, metrics, HODA benchmark). |

## Dataset details

- **Size:** 2,560 images, digits 0–9.
- **Resolution:** 16×16, grayscale, **white background / dark ink**.
- **Class distribution** (imbalanced — digits 2 and 3 were over-sampled):

  | digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | count | 187 | 196 | 533 | 533 | 192 | 187 | 187 | 182 | 182 | 181 |

- **Writer metadata:** ~84% of images carry a writer label (eight labelled writers), enabling
  **writer-disjoint** evaluation. The remaining images are marked `unknown`. *(Note: the labels
  `m` and `ma` may refer to the same writer; merge them if so.)*

## Quick start

```python
from load_persian_digits import load, writer_disjoint_split
X, y, writers = load("PersianDigits.npz")          # X in [0,1], shape (N,1,16,16)
(Xtr, ytr), (Xte, yte) = writer_disjoint_split(X, y, writers, test_size=0.2, seed=0)
```

Run the trained model on an image:

```bash
pip install -r requirements.txt
python predict.py path/to/a_digit.jpg
```

## Model & results

A small CNN (two conv blocks + a classifier head, ~151k parameters; see `model.py`).
Evaluated with a **frozen held-out test set** and **5-fold cross-validation × 3 seeds**,
with normalization/PCA fit on training folds only (no leakage):

- **Cross-validation accuracy:** 97.1% ± 0.8%
- **Frozen test-set accuracy:** 96.5% (macro-F1 96.2%)
- **Sanity check on HODA** (standard benchmark), same pipeline: ~99%

See the notebook for per-class metrics, confusion matrices, the classical baselines
(SVM, Random Forest, KNN, Decision Tree, Naive Bayes), and the cross-dataset comparison.

## Reproducing

Open `persian_digits_rigorous_cnn.ipynb`, set `DATA_DIR` to the `dataset` folder, and run
top to bottom. Set `USE_WRITER_DISJOINT = True` for the writer-disjoint protocol.

## License

Released under **Creative Commons Attribution 4.0 International (CC BY 4.0)** — free to use,
share, and adapt with attribution. See `LICENSE`. If you use this dataset or model, please
cite the paper (see `CITATION.cff`).

## Authors

Dorna Jokar, Morteza Khorsandnikoo (corresponding author), Negin Aani, Alireza Salarkia,
Aila Sheidaei, Hanieh Ziaratian.
