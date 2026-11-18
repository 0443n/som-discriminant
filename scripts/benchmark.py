"""Benchmark KohonenDA v1/v2 against classical and non-parametric classifiers.

Experiments
-----------
clean          : all datasets in kohonen_da.datasets, repeated stratified 68/32 splits,
                 StandardScaler fitted on the training part.
contamination  : a fraction eps of *training* objects get gross errors
                 (x += N(0, (5 * feature_std)^2), labels kept); tested on clean data.
                 RobustScaler (median/IQR) for every method, so that the experiment
                 measures the classifiers, not the breakdown of the scaler.

All hyper-parameters are fixed defaults (no tuning for any method); SOMs are 3x3 per
class as in the paper. v2_joint_tNN = v2 joint with trim=0.NN (outlier-resistant maps).
v3_* = KohonenDA3 (neuron-level diag mixtures), see kohonen_da/v3.py; *_tauto = adaptive
trimming; v3_auto* = KohonenDA3Auto (map size and scoring chosen out-of-fold).

Usage:
  python scripts/benchmark.py clean [--runs 50]
  python scripts/benchmark.py contamination [--runs 30]
Writes results/benchmark/<experiment>.csv and prints summary tables.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from kohonen_da import KohonenDA, KohonenDA2, KohonenDA3, KohonenDA3Auto
from kohonen_da.baselines import GMMClassifier
from kohonen_da.datasets import ALL, REAL, SYNTHETIC

OUT = Path(__file__).resolve().parent.parent / "results" / "benchmark"
SOM_METHODS = ["v1", "nearest_map", "v2_pvalue", "v2_joint", "v2_joint_t10", "v2_joint_t20",
               "v3_density", "v3_joint_density", "v3_joint_sorted", "v3_density_t10", "v3_density_t20",
               "v3_density_tauto", "v3_auto", "v3_auto_trim"]
BASELINES = ["lda", "qda", "gnb", "gmm3", "knn5", "svm_rbf", "rf"]
METHODS = SOM_METHODS + BASELINES
CONTAM_DATASETS = ["iris", "wine", "breast_cancer", "multimodal", "rings"]
EPS_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]


def baseline(name, seed):
    return {
        "lda": lambda: LinearDiscriminantAnalysis(),
        "qda": lambda: QuadraticDiscriminantAnalysis(reg_param=1e-2),
        "gnb": lambda: GaussianNB(),
        "gmm3": lambda: GMMClassifier(3, random_state=seed),
        "knn5": lambda: KNeighborsClassifier(5),
        "svm_rbf": lambda: SVC(),
        "rf": lambda: RandomForestClassifier(200, random_state=seed),
    }[name]()


def contaminate(X, eps, rng):
    X = X.copy()
    m = int(round(eps * len(X)))
    if m:
        idx = rng.choice(len(X), m, replace=False)
        X[idx] += rng.normal(size=(m, X.shape[1])) * 5 * X.std(0)
    return X


def run_split(dataset, seed, eps=0.0, scaler="standard"):
    warnings.filterwarnings("ignore")
    X, y = ALL[dataset](seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.32, stratify=y, random_state=seed)
    Xtr = contaminate(Xtr, eps, np.random.default_rng(10_000 + seed))
    sc = (StandardScaler() if scaler == "standard" else RobustScaler()).fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    rows = []

    def add(method, pred, t, choice=""):
        rows.append(dict(dataset=dataset, seed=seed, eps=eps, method=method,
                         acc=float((pred == yte).mean()), fit_s=t, choice=choice))

    t0 = time.perf_counter()
    v1 = KohonenDA(random_state=seed).fit(Xtr, ytr)  # histograms on train (no leakage)
    t = time.perf_counter() - t0
    add("v1", v1.predict(Xte), t)
    add("nearest_map", v1.classes_[v1.bmu_distances(Xte).argmin(1)], t)

    t0 = time.perf_counter()
    v2 = KohonenDA2(random_state=seed).fit(Xtr, ytr)
    t = time.perf_counter() - t0
    add("v2_pvalue", v2.predict(Xte), t)
    v2.set_params(scoring="joint")
    add("v2_joint", v2.predict(Xte), t)

    for trim, name in [(0.1, "v2_joint_t10"), (0.2, "v2_joint_t20")]:
        t0 = time.perf_counter()
        vt = KohonenDA2(scoring="joint", trim=trim, random_state=seed).fit(Xtr, ytr)
        add(name, vt.predict(Xte), time.perf_counter() - t0)

    t0 = time.perf_counter()
    v3 = KohonenDA3(covariance="diag", random_state=seed).fit(Xtr, ytr)
    t = time.perf_counter() - t0
    for scoring in ("density", "joint_density", "joint_sorted"):
        v3.set_params(scoring=scoring)
        add(f"v3_{scoring}", v3.predict(Xte), t)
    for trim, name in [(0.1, "v3_density_t10"), (0.2, "v3_density_t20")]:
        t0 = time.perf_counter()
        v3t = KohonenDA3(covariance="diag", trim=trim, random_state=seed).fit(Xtr, ytr)
        add(name, v3t.predict(Xte), time.perf_counter() - t0)

    t0 = time.perf_counter()
    v3a = KohonenDA3(covariance="diag", trim="auto", random_state=seed).fit(Xtr, ytr)
    add("v3_density_tauto", v3a.predict(Xte), time.perf_counter() - t0)
    for trim, name in [(0.0, "v3_auto"), ("auto", "v3_auto_trim")]:
        t0 = time.perf_counter()
        au = KohonenDA3Auto(trim=trim, random_state=seed).fit(Xtr, ytr)
        add(name, au.predict(Xte), time.perf_counter() - t0, f"{au.map_size_}x{au.map_size_}/{au.scoring_}")

    for name in BASELINES:
        t0 = time.perf_counter()
        est = baseline(name, seed).fit(Xtr, ytr)
        add(name, est.predict(Xte), time.perf_counter() - t0)
    return rows


def run(tasks, n_jobs):
    out = Parallel(n_jobs=n_jobs, verbose=0)(delayed(run_split)(*t) for t in tasks)
    return pd.DataFrame([r for rows in out for r in rows])


# ------------------------------------------------------------------ reporting
def summarise_clean(df, ref="v3_density"):
    acc = df.pivot_table(index="dataset", columns="method", values="acc", aggfunc="mean")
    acc = acc.loc[[d for d in ALL if d in acc.index], METHODS]
    ranks = acc.rank(axis=1, ascending=False)
    print("\nMean accuracy (rows: datasets)")
    print(acc.round(3).to_string())
    for group, names in [("real", REAL), ("synthetic", SYNTHETIC), ("all", ALL)]:
        r = ranks.loc[[d for d in names if d in ranks.index]].mean()
        print(f"\nAverage rank, {group} datasets (1 = best):")
        print(r.sort_values().round(2).to_string())

    fr = stats.friedmanchisquare(*[acc[m] for m in METHODS])
    print(f"\nFriedman test over datasets: chi2={fr.statistic:.2f}, p={fr.pvalue:.2e}")

    print(f"\n{ref} vs each method: paired Wilcoxon over splits per dataset (p<0.05)")
    rows = []
    for m in METHODS:
        if m == ref:
            continue
        better = worse = 0
        for d in acc.index:
            sub = df[df.dataset == d].pivot_table(index="seed", columns="method", values="acc")
            diff = sub[ref] - sub[m]
            if (diff != 0).any():
                p = stats.wilcoxon(sub[ref], sub[m], zero_method="zsplit").pvalue
                if p < 0.05:
                    better += diff.mean() > 0
                    worse += diff.mean() < 0
        rows.append(dict(method=m, ref_sig_better=better, ref_sig_worse=worse,
                         ties=len(acc.index) - better - worse))
    print(pd.DataFrame(rows).to_string(index=False))

    t = df.groupby("method").fit_s.mean().loc[METHODS]
    print("\nMean fit time per split, s:")
    print(t.round(3).to_string())
    if "choice" in df.columns:
        ch = df[df.method == "v3_auto"].groupby("dataset").choice.agg(
            lambda c: ", ".join(f"{k} {v}" for k, v in c.value_counts().head(3).items()))
        print("\nv3_auto choices (most frequent first):")
        print(ch.loc[[d for d in ALL if d in ch.index]].to_string())
    return acc, ranks


def summarise_contamination(df):
    acc = df.pivot_table(index=["dataset", "eps"], columns="method", values="acc",
                         aggfunc="mean")[METHODS]
    print("\nMean accuracy under training-set contamination")
    print(acc.round(3).to_string())
    base = acc.xs(0.0, level="eps")
    drop = (acc - base.reindex(acc.index, level="dataset")).groupby(level="eps").mean()
    print("\nMean accuracy change vs. eps=0, averaged over datasets")
    print(drop.round(3).to_string())
    worst = drop.loc[max(EPS_LEVELS)].sort_values(ascending=False)
    print(f"\nRobustness ranking at eps={max(EPS_LEVELS)} (smallest drop first):")
    print(worst.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["clean", "contamination", "report"])
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if a.experiment == "clean":
        runs = a.runs or 50
        df = run([(d, s) for d in ALL for s in range(runs)], a.jobs)
        df.to_csv(OUT / "clean.csv", index=False)
        summarise_clean(df)
    elif a.experiment == "contamination":
        runs = a.runs or 30
        tasks = [(d, s, e, "robust") for d in CONTAM_DATASETS for e in EPS_LEVELS
                 for s in range(runs)]
        df = run(tasks, a.jobs)
        df.to_csv(OUT / "contamination.csv", index=False)
        summarise_contamination(df)
    else:  # re-print summaries from saved CSVs
        if (OUT / "clean.csv").exists():
            summarise_clean(pd.read_csv(OUT / "clean.csv"))
        if (OUT / "contamination.csv").exists():
            summarise_contamination(pd.read_csv(OUT / "contamination.csv"))
    print(f"\n[{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
