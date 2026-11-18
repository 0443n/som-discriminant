# kohonen-da

Reproduction (and, later, extensions) of

> Komarov I.V., Kuravsky L.S. **Discriminant Analysis Based on Kohonen Statistics.**
> *Modelling and Data Analysis*, 2023, 13(4), 176–182. doi:[10.17759/mda.2023130411](https://doi.org/10.17759/mda.2023130411)

The original 2023 code is lost; `kohonen_da/` is a clean-room reimplementation from the paper text.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'   # exact versions: requirements.lock
.venv/bin/python -m pytest
.venv/bin/python scripts/reproduce_paper.py --runs 200        # ~1 min, writes results/
```

## Method (as implemented)

`KohonenDA` (scikit-learn estimator, `kohonen_da/classifier.py`):

1. One MiniSom map per class, trained only on that class; weights initialised with random samples.
2. For calibration samples of class *c*: Euclidean distance to the BMU on map *c*.
3. Histogram with `1 + floor(log2 N)` bins (Sturges), frequencies normalised to sum to 1.
4. New object: bin frequency of `d_c(x)` in histogram *c* for every class → normalise → argmax.

### Assumptions where the paper is silent

| Item | Choice | Why |
|---|---|---|
| Feature scaling | none (raw cm) | Only raw features reproduce the distance ranges of Fig. 3 (setosa ≈0.1–0.5, versicolor ≈0.2–0.6, virginica ≈0.2–0.9); standardised/min-max don't |
| Split | 34 train / 16 test per class | Confusion matrix sums to 48 = 16×3 |
| SOM hyper-params | σ=1.0, lr=0.5, Gaussian, 1000 random-order iterations | MiniSom defaults; not reported |
| All classes get probability 0 | fall back to nearest map (`zero_fallback="nearest"`) | Case not covered by the paper |
| Calibration data | the test sample (protocol `paper`) | Literal reading: "a test sample of each a-priori class is used" for the histograms, and the confusion matrix is on the test sample |

## Results (200 random stratified splits, 3×3 maps)

| Protocol | Histograms built on | Mean acc. | Errors / 48 | Undecided / 48 |
|---|---|---|---|---|
| `paper` | test set (= evaluation set) | 0.964 ± 0.026 | 1.72 | 0.00 |
| `train_cal` | training set | 0.949 ± 0.026 | 2.46 | 6.94 |
| `three_way` | separate calibration split (smaller train) | 0.922 ± 0.039 | 3.76 | 5.78 |
| nearest map only (no histograms) | — | 0.958 | 2.03 | — |
| `train_cal` **without** fallback | training set | 0.813 | 8.96 | counted as errors |
| Fisher LDA | — | 0.978 ± 0.019 | 1.08 | — |

*Undecided* = every class's histogram gives probability 0 (the distance falls outside every histogram's range).

Paired comparison with LDA on identical splits:
- `paper` protocol: better 20.5%, equal 26.5%, worse 53.0%
- `train_cal`: better 5.5%, equal 19.0%, worse 75.5%

Single illustrative run (seed 0) gives 47/48 vs LDA 48/48; the paper's 46/48 is within the typical spread.

### What this shows

1. **The published number is reproducible** (≈1.7 errors / 48 on average vs. 2 in the paper).
2. **The paper protocol leaks.** Histograms are fitted on the very objects being classified. By construction, each test object then falls inside its own class's histogram range, in a bin of frequency ≥ 1/16. That is why the paper never runs into the zero-probability problem.
3. **Without leakage, the zero-probability flaw dominates.** About 7 of 48 test objects get probability 0 for *every* class, and about 8 get 0 for their own class. Using the histogram rule alone gives 81% accuracy. The method only reaches 95% because of the nearest-map fallback, which isn't part of the paper.
4. **The histogram step currently adds nothing.** Plain "nearest map wins" (0.958) is as good as or better than the honest histogram variants.
5. **The claim "strictly no worse than Fisher LDA" does not hold.** On iris, LDA is better on average and on most paired splits under every protocol.

These are the targets for the improved method, implemented below as v2.

## v2 (`KohonenDA2`, `kohonen_da/v2.py`)

| 2023 weakness | v2 fix |
|---|---|
| Bin lookup: zero outside the histogram range, bin mass ≠ density, not comparable across classes | **Class-conditional conformal p-value** of the BMU distance: `p_c(x) = (1 + #{calib d ≥ d_c(x)}) / (n_c + 1)`. Never 0, monotone, no bins, and uniform under class *c*, so comparable across classes |
| Histograms fitted on the evaluation set | **Cross-fitted calibration**: out-of-fold BMU distances (5 folds); final maps refit on all data |
| Compares p(d₁\|1), p(d₂\|2), … (different variables) | Optional **`scoring="joint"`**: logistic regression on log distances to *all* maps → P(c \| d₁…d_K) |
| No "none of the above" | **Reject option** (`reject_alpha`) and **prediction sets**: P(true class excluded) ≤ α |

Run: `.venv/bin/python scripts/evaluate_v2.py [--dataset iris|wine|breast_cancer] [--scale]`

### Closed-set accuracy (same splits for all methods; test = 32%)

| Method | iris (200 splits, raw) | wine (100 splits, standardised) |
|---|---|---|
| v1, paper protocol (leaky) | 0.964 | 0.929 |
| v1, honest (train-cal + nearest-map fallback) | 0.949 | 0.898 |
| nearest map only | 0.958 | 0.947 |
| **v2 `pvalue`** | 0.957 | 0.936 |
| **v2 `joint`** | 0.954 | **0.973** |
| Fisher LDA | 0.978 | 0.982 |
| QDA | 0.976 | 0.989 |
| kNN (k=5) | 0.963 | 0.959 |

### Conformal coverage: P(p_true < α) on test objects (should be ≤ α)

| α | iris | wine |
|---|---|---|
| 0.05 | 0.027 | 0.037 |
| 0.10 | 0.077 | 0.074 |
| 0.20 | 0.163 | 0.170 |

### Open set: train on 2 classes, then show held-out known objects plus all objects of the unseen class

Each cell shows the false-reject rate on known classes / the detection rate on the unseen class.

| Unseen class | v2 α=0.05 | v2 α=0.20 | v1 (all-zero = reject) | v2 AUROC |
|---|---|---|---|---|
| iris setosa | 0.017 / 1.000 | 0.132 / 1.000 | 0.141 / 1.000 | 0.992 |
| iris versicolor | 0.024 / 0.336 | 0.154 / 0.836 | 0.146 / 0.749 | 0.914 |
| iris virginica | 0.020 / 0.746 | 0.146 / 0.885 | 0.159 / 0.888 | 0.940 |
| wine 0 | 0.025 / 0.401 | 0.167 / 0.902 | 0.148 / 0.776 | 0.937 |
| wine 1 | 0.039 / 0.723 | 0.161 / 0.961 | 0.203 / 0.965 | 0.969 |
| wine 2 | 0.033 / 0.468 | 0.142 / 0.821 | 0.107 / 0.701 | 0.927 |

### Takeaways so far

- v2 beats the *honest* 2023 method everywhere and never produces undefined outputs.
- `joint` scoring is the big accuracy gain on wine: 1.5 errors vs 5.8 for honest v1, against LDA's 1.0.
- `pvalue` scoring ≈ nearest map for accuracy; its value lies in calibrated typicality, the reject option and prediction sets.
- Coverage holds, slightly conservatively. The likely cause is that the final maps are trained on more data than the fold maps.
- Open set: v2 lets you *choose* the false-reject rate (v1's is an accident of histogram ranges). At comparable false-reject rates it detects unseen classes as well or better.
- On these near-Gaussian datasets LDA/QDA are still best. The method's niche has to be shown on non-Gaussian, multimodal or outlier-contaminated data (next step).

## Layout

```
kohonen_da/classifier.py     KohonenDA estimator (paper-faithful, v1)
kohonen_da/v2.py             KohonenDA2 (conformal, cross-fitted, reject option)
kohonen_da/som.py            shared SOM training / BMU distances
scripts/reproduce_paper.py   Section 4 reproduction + repeated-split study
scripts/evaluate_v2.py       v2 vs v1 vs baselines: accuracy, coverage, open set
tests/                       unit tests
results/                     figures and per-split CSVs
```
