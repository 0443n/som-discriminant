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

## Benchmark (`scripts/benchmark.py`)

Setup:
- **Datasets:** 4 real (iris, wine, breast cancer, digits) and 5 synthetic non-Gaussian (`kohonen_da/datasets.py`). The synthetic sets are multimodal (3 blobs per class), rings, moons, XOR and heavy-tailed (Student-t, df=1.5). Each synthetic set has 2 informative dimensions plus 2 pure-noise dimensions.
- **Splits:** 50 repeated stratified 68/32 splits, with features standardised on the training part.
- **Hyper-parameters:** library defaults for every method, with no tuning anywhere. SOMs are 3×3 per class, as in the paper.
- **`gmm3`:** a per-class 3-component Gaussian mixture with the Bayes rule (`kohonen_da/baselines.py`). It is the parametric counterpart of "one SOM per class".

```bash
.venv/bin/python scripts/benchmark.py clean           # ~2 min on 16 cores
.venv/bin/python scripts/benchmark.py contamination   # ~3 min
.venv/bin/python scripts/plot_benchmark.py
```

### Clean data: mean accuracy

| dataset | v1 | nearest map | v2 pvalue | v2 joint | LDA | QDA | GNB | GMM3 | kNN5 | SVM-RBF | RF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iris | .934 | .944 | .947 | .952 | **.978** | **.978** | .952 | .959 | .948 | .957 | .948 |
| wine | .895 | .947 | .928 | .969 | .981 | **.992** | .973 | .837 | .956 | .983 | .983 |
| breast cancer | .861 | .952 | .874 | .953 | .957 | .962 | .933 | .953 | .965 | **.973** | .957 |
| digits | .751 | .951 | .904 | .950 | .951 | .967 | .789 | .850 | .971 | **.979** | .974 |
| multimodal | .794 | .855 | .838 | .854 | .627 | .896 | .725 | .958 | .919 | .934 | **.976** |
| rings | .512 | .609 | .557 | .756 | .354 | .751 | .804 | .769 | .673 | .910 | **.939** |
| moons | .803 | .843 | .843 | .843 | .857 | .852 | .855 | .881 | .880 | .865 | **.907** |
| xor | .875 | .909 | .901 | .904 | .506 | **.959** | .508 | .945 | .935 | .935 | .931 |
| heavy-tailed | .441 | .650 | .637 | .660 | .707 | .522 | .466 | .670 | .724 | **.738** | **.738** |

Average rank over all 9 datasets, where 1 is best: SVM 2.56, RF 2.72, QDA 4.17, kNN 4.50, GMM3 5.00, **v2 joint 6.39**, LDA 6.39, nearest map 7.56, GNB 7.61, v2 pvalue 8.78, **v1 10.33**. The Friedman test gives χ²=50.0, p=2.6e-7.

v2 joint against each method, using a paired Wilcoxon test over splits within each dataset (p<0.05). Each cell counts datasets:

| | v1 | nearest map | v2 pvalue | LDA | QDA | GNB | GMM3 | kNN5 | SVM | RF |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 joint significantly better | 9 | 3 | 6 | 3 | 1 | 5 | 2 | 2 | 0 | 0 |
| v2 joint significantly worse | 0 | 0 | 0 | 4 | 6 | 1 | 3 | 6 | 8 | 7 |

Mean fit time per split: v2 2.6 s (5-fold cross-fit, i.e. 6× SOM training), v1 0.45 s, RF 1.2 s, the rest < 0.11 s.

### Training-set contamination (gross errors, clean test set)

A fraction ε of the training objects get `x += N(0, (5·std)²)` with labels kept. For all methods the features go through a RobustScaler fitted on the contaminated training set, so the experiment measures the classifiers rather than the scaler breaking down. There are 30 splits per (dataset, ε).

![contamination](results/benchmark/contamination.png)

Mean accuracy change vs. clean data, averaged over iris, wine, breast cancer, multimodal and rings:

| ε | v1 | nearest map | v2 pvalue | v2 joint | LDA | QDA | GNB | GMM3 | kNN5 | SVM-RBF | RF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | −.027 | −.012 | −.017 | −.016 | −.025 | −.147 | −.113 | −.039 | −.001 | −.041 | −.001 |
| 0.10 | −.062 | −.031 | −.039 | −.034 | −.044 | −.201 | −.181 | −.098 | −.003 | −.082 | −.002 |
| 0.20 | −.092 | −.049 | −.054 | −.057 | −.062 | −.294 | −.281 | −.176 | −.008 | −.138 | −.004 |
| 0.30 | −.103 | −.074 | −.086 | −.098 | −.070 | −.309 | −.306 | −.222 | −.011 | −.164 | −.008 |

Full per-dataset tables: `python scripts/benchmark.py report`.

### Ablation: map size and neighbourhood width (v2, 20 splits)

My hypothesis was that neighbourhood smoothing (σ) pulls prototypes between modes. **It was rejected:** σ=0.3 vs 1.0 changes accuracy by ≤ 0.02. A larger map helps modestly. Going from 3×3 to 4×4 gives (joint scoring) multimodal .859→.888, rings .754→.787, digits .951→.961 and wine .972→.978. That does not close the gap to GMM3 or kNN on multimodal.

### What the benchmark says, honestly

1. **v2 fixes v1.** It is significantly better on 9/9 datasets, and its outputs are always defined. On digits v1 collapses to .75, because in 64-D most test points fall outside every histogram.
2. **Accuracy is not the method's selling point.** v2 joint is mid-pack, tied with LDA on average rank. SVM and RF beat it on almost every dataset, and kNN/QDA on most. On near-Gaussian real data LDA/QDA win. On nonlinear data v2 beats LDA by a wide margin (multimodal, rings, XOR), but flexible methods beat v2.
3. **The 2023 "robust to outliers" claim is half true.** The SOM methods degrade about as much as LDA. They are far more robust than density-based QDA/GNB/GMM and than SVM, but far *less* robust than kNN and RF, which barely move.
4. **Where the method genuinely adds value** is calibrated per-class typicality: the conformal coverage guarantee, prediction sets, and the reject option for unseen classes (see v2 section). Those are things SVM/RF/kNN don't offer out of the box. A follow-up paper should be framed around this, not around accuracy.

## Layout

```
kohonen_da/classifier.py     KohonenDA estimator (paper-faithful, v1)
kohonen_da/v2.py             KohonenDA2 (conformal, cross-fitted, reject option)
kohonen_da/som.py            shared SOM training / BMU distances
scripts/reproduce_paper.py   Section 4 reproduction + repeated-split study
scripts/evaluate_v2.py       v2 vs v1 vs baselines: accuracy, coverage, open set
scripts/benchmark.py         9-dataset benchmark + contamination experiment
scripts/plot_benchmark.py    contamination figure
kohonen_da/datasets.py       real + synthetic benchmark datasets
kohonen_da/baselines.py      per-class GMM classifier
tests/                       unit tests
results/                     figures and per-split CSVs
```
