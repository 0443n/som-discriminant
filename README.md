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
| No "none of the above" | **Reject option** (`reject_alpha`) and **prediction sets**: P(true class excluded) ≤ α. `reject_mode="pooled"` (default) uses a conformal p-value of the distance to the nearest prototype over all maps; `"classwise"` rejects when every class p-value is < α |
| Prototypes pulled by gross outliers (added after the benchmark) | **`trim`**: fit, drop the `trim` share of each class with the largest BMU distance, re-initialise and refit |

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

Each cell shows the false-reject rate on known classes / the detection rate on the unseen class. This early experiment uses `reject_mode="classwise"`; the full open-set benchmark below shows that `"pooled"` is better.

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
- On these near-Gaussian datasets LDA/QDA are still best. See the benchmark below for non-Gaussian and contaminated data.

## Benchmark (`scripts/benchmark.py`, `scripts/openset.py`)

Setup:
- **Datasets:** 4 real (iris, wine, breast cancer, digits) and 5 synthetic non-Gaussian (`kohonen_da/datasets.py`). The synthetic sets are multimodal (3 blobs per class), rings, moons, XOR and heavy-tailed (Student-t, df=1.5). Each synthetic set has 2 informative dimensions plus 2 pure-noise dimensions.
- **Splits:** 50 repeated stratified 68/32 splits, with features standardised on the training part.
- **Hyper-parameters:** library defaults for every method, with no tuning anywhere. SOMs are 3×3 per class, as in the paper.
- **`gmm3`:** a per-class 3-component Gaussian mixture with the Bayes rule (`kohonen_da/baselines.py`). It is the parametric counterpart of "one SOM per class".
- **`v2 joint tNN`:** v2 joint with `trim=0.NN`.

```bash
.venv/bin/python scripts/benchmark.py clean           # ~7 min on 16 cores
.venv/bin/python scripts/benchmark.py contamination   # ~8 min
.venv/bin/python scripts/openset.py                   # ~3 min
.venv/bin/python scripts/plot_benchmark.py
.venv/bin/python scripts/benchmark.py report          # re-print tables from saved CSVs
.venv/bin/python scripts/openset.py --report
```

### Clean data: mean accuracy

| dataset | v1 | nearest map | v2 pvalue | v2 joint | v2 joint t10 | v2 joint t20 | LDA | QDA | GNB | GMM3 | kNN5 | SVM-RBF | RF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iris | .934 | .944 | .947 | .952 | .952 | .949 | **.978** | **.978** | .952 | .959 | .948 | .957 | .948 |
| wine | .895 | .947 | .928 | .969 | .970 | .972 | .981 | **.992** | .973 | .837 | .956 | .983 | .983 |
| breast cancer | .861 | .952 | .874 | .953 | .950 | .949 | .957 | .962 | .933 | .953 | .965 | **.973** | .957 |
| digits | .751 | .951 | .904 | .950 | .953 | .950 | .951 | .967 | .789 | .850 | .971 | **.979** | .974 |
| multimodal | .794 | .855 | .838 | .854 | .847 | .850 | .627 | .896 | .725 | .958 | .919 | .934 | **.976** |
| rings | .512 | .609 | .557 | .756 | .750 | .729 | .354 | .751 | .804 | .769 | .673 | .910 | **.939** |
| moons | .803 | .843 | .843 | .843 | .848 | .844 | .857 | .852 | .855 | .881 | .880 | .865 | **.907** |
| xor | .875 | .909 | .901 | .904 | .909 | .907 | .506 | **.959** | .508 | .945 | .935 | .935 | .931 |
| heavy-tailed | .441 | .650 | .637 | .660 | .678 | .685 | .707 | .522 | .466 | .670 | .724 | **.738** | **.738** |

Average rank over all 9 datasets, where 1 is best: SVM 2.56, RF 2.94, QDA 4.39, kNN 5.17, GMM3 5.67, **v2 joint t10 7.00**, LDA 7.17, v2 joint 7.50, v2 joint t20 7.89, nearest map 8.78, GNB 8.83, v2 pvalue 10.78, **v1 12.33**. The Friedman test gives χ²=58.5, p=4.2e-8.

v2 joint against each method, using a paired Wilcoxon test over splits within each dataset (p<0.05). Each cell counts datasets:

| | v1 | nearest map | v2 pvalue | t10 | t20 | LDA | QDA | GNB | GMM3 | kNN5 | SVM | RF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 joint significantly better | 9 | 3 | 6 | 0 | 1 | 3 | 1 | 5 | 2 | 2 | 0 | 0 |
| v2 joint significantly worse | 0 | 0 | 0 | 2 | 1 | 4 | 6 | 1 | 3 | 6 | 8 | 7 |

Mean fit time per split: v2 2.5 s (5-fold cross-fit, i.e. 6× SOM training), v2 trimmed 4.7–4.9 s, v1 0.44 s, RF 1.1 s, the rest < 0.1 s.

### Training-set contamination (gross errors, clean test set)

A fraction ε of the training objects get `x += N(0, (5·std)²)` with labels kept. For all methods the features go through a RobustScaler fitted on the contaminated training set, so the experiment measures the classifiers rather than the scaler breaking down. There are 30 splits per (dataset, ε).

![contamination](results/benchmark/contamination.png)

Mean accuracy change vs. clean data, averaged over iris, wine, breast cancer, multimodal and rings:

| ε | v1 | nearest map | v2 pvalue | v2 joint | v2 joint t10 | v2 joint t20 | LDA | QDA | GNB | GMM3 | kNN5 | SVM-RBF | RF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | −.027 | −.012 | −.017 | −.016 | −.006 | −.006 | −.025 | −.147 | −.113 | −.039 | −.001 | −.041 | −.001 |
| 0.10 | −.062 | −.031 | −.039 | −.034 | −.010 | −.010 | −.044 | −.201 | −.181 | −.098 | −.003 | −.082 | −.002 |
| 0.20 | −.092 | −.049 | −.054 | −.057 | −.028 | −.012 | −.062 | −.294 | −.281 | −.176 | −.008 | −.138 | −.004 |
| 0.30 | −.103 | −.074 | −.086 | −.098 | −.059 | **−.035** | −.070 | −.309 | −.306 | −.222 | −.011 | −.164 | −.008 |

### Open-set recognition (`scripts/openset.py`)

- **Protocol:** use every dataset with ≥ 3 classes. Each class in turn is "unseen". Train on the rest (68/32 split of the known objects), then score the held-out known objects and all unseen ones. There are 20 splits per held-out class.
- **"Unknown" scores:**
  - v2 class-wise (−max_c p_c) and v2 pooled (−novelty p-value of min_c d_c);
  - the raw nearest-map distance;
  - the mean distance to the 5 nearest training objects;
  - the per-class GMM log-likelihood;
  - the Mahalanobis distance to the nearest class mean (shared covariance);
  - −max posterior probability of LDA, SVM and RF.

AUROC (unseen vs. known):

| dataset | v2 class-wise | **v2 pooled** | nearest map | kNN dist | GMM loglik | Mahalanobis | LDA maxprob | SVM maxprob | RF maxprob |
|---|---|---|---|---|---|---|---|---|---|
| iris | .934 | .956 | .960 | .952 | .958 | **.985** | .393 | .947 | .563 |
| wine | .947 | .965 | .967 | .968 | .866 | **.977** | .684 | .868 | .854 |
| digits | .899 | .909 | .909 | .927 | .873 | .870 | .849 | .960 | **.965** |
| multimodal | .834 | .838 | .837 | .885 | **.954** | .719 | .281 | .683 | .797 |
| rings | .604 | .626 | .626 | .711 | **.725** | .545 | .487 | .367 | .521 |
| heavy-tailed | .636 | .639 | .639 | .678 | .670 | .668 | .587 | .690 | **.712** |

Average AUROC rank: kNN dist 2.83, nearest map 3.83, GMM 3.83, **v2 pooled 4.33**, Mahalanobis 4.67, RF 5.17, SVM 5.50, v2 class-wise 6.00, LDA 8.83. v2 pooled is significantly better (paired Wilcoxon, p<0.05) than v2 class-wise on 5/6 datasets and than LDA on 6/6. kNN distance is significantly better than it on 5/6.

The SVM/RF/kNN/GMM baselines have no built-in way to pick a threshold. v2 does: α=0.05 is fixed in advance, with no oracle. Cells show false-reject rate / detection rate:

| dataset | v2 class-wise | v2 pooled |
|---|---|---|
| iris | .020 / .573 | .034 / .681 |
| wine | .030 / .520 | .035 / .722 |
| digits | .024 / .134 | .047 / .383 |
| multimodal | .034 / .148 | .044 / .199 |
| rings | .023 / .051 | .048 / .131 |
| heavy-tailed | .022 / .027 | .039 / .044 |

### Ablation: map size and neighbourhood width (v2, 20 splits)

My hypothesis was that neighbourhood smoothing (σ) pulls prototypes between modes. **It was rejected:** σ=0.3 vs 1.0 changes accuracy by ≤ 0.02. A larger map helps modestly. Going from 3×3 to 4×4 gives (joint scoring) multimodal .859→.888, rings .754→.787, digits .951→.961 and wine .972→.978. That does not close the gap to GMM3 or kNN on multimodal.

### What the benchmark says, honestly

1. **v2 fixes v1.** It is significantly better on 9/9 datasets, and its outputs are always defined. On digits v1 collapses to .75, because in 64-D most test points fall outside every histogram.
2. **Accuracy is not the method's selling point.** v2 joint is mid-pack, about level with LDA on average rank. SVM and RF beat it on almost every dataset, and kNN/QDA on most. On nonlinear data it beats LDA by a wide margin, but flexible methods beat it.
3. **Robustness: v1 is middling, trimmed v2 is good.**
   - Untrimmed SOM methods degrade about as much as LDA.
   - With `trim=0.2`, v2 loses only 3.5 points at ε=0.3 (untrimmed: 9.8). That makes it the 3rd most robust method, behind RF and kNN, which barely move, and ahead of LDA, SVM and all density-based methods.
   - On clean data trimming costs nothing measurable.
4. **Open set: the guarantee is v2's real asset, its ranking is middling.**
   - The pooled score holds the false-reject rate at ≤ α out of the box (3.4–4.8% at α=5%).
   - Its ranking quality is mid-pack. The kNN distance beats it, and so does GMM on multimodal data and SVM/RF on digits.
   - The per-class p-value is a *worse* novelty score than the pooled distance. Normalising per class inflates the p-values for classes with a broad spread.
5. **Implication for a paper.** The defensible contribution is a careful, fully calibrated version of the 2023 idea: per-class prototype maps, cross-fitted conformal p-values, prediction sets, pooled novelty detection with a guaranteed false-reject rate, and trimmed training for robustness. It should be presented with an honest comparison, not as a more accurate classifier.
   - Caveat: the conformal wrapper is generic. Any score, including the kNN distance, could be calibrated the same way. Novelty therefore has to come from the combination plus the interpretability of a small set of per-class prototypes, not from the guarantee alone.

## Layout

```
kohonen_da/classifier.py     KohonenDA estimator (paper-faithful, v1)
kohonen_da/v2.py             KohonenDA2 (conformal, cross-fitted, reject option)
kohonen_da/som.py            shared SOM training / BMU distances
scripts/reproduce_paper.py   Section 4 reproduction + repeated-split study
scripts/evaluate_v2.py       v2 vs v1 vs baselines: accuracy, coverage, open set
scripts/benchmark.py         9-dataset benchmark + contamination experiment
scripts/openset.py           open-set recognition benchmark
scripts/plot_benchmark.py    contamination figure
kohonen_da/datasets.py       real + synthetic benchmark datasets
kohonen_da/baselines.py      per-class GMM classifier
tests/                       unit tests
results/                     figures and per-split CSVs
```
