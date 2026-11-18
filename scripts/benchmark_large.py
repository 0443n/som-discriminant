"""Benchmark on larger OpenML datasets (kohonen_da/large_datasets.py).

None of these datasets were used while developing v2/v3, so this doubles as a
held-out check of the design decisions.

clean   : 5 stratified 68/32 splits per dataset, StandardScaler on the training part.
          SOM methods: v1 and v2 (3x3, as in the paper), v3 density with 3x3 maps and
          the recommended KohonenDA3Auto(trim="auto") with map sizes {3,5,8,12}
          (larger maps matter once classes have thousands of objects).
openset : 5 splits per dataset; in each, a random 30% of the classes (at least one) is
          unseen: train on the rest, score held-out known objects + all unseen ones.

Usage: python -I scripts/benchmark_large.py clean|openset|report [--runs 5] [--jobs 12]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openset import det_at, mahalanobis_score  # noqa: E402

from kohonen_da import KohonenDA, KohonenDA2, KohonenDA3, KohonenDA3Auto  # noqa: E402
from kohonen_da import large_datasets as L  # noqa: E402
from kohonen_da.baselines import GMMClassifier  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "benchmark_large"
# heaviest first, for load balancing
DATASETS = ["letter", "mnist_pca50", "covertype", "pendigits", "optdigits", "satimage", "segment"]
METHODS = ["v1", "v2_joint", "v3_density_3x3", "v3_auto_trim",
           "lda", "qda", "gnb", "gmm3", "knn5", "svm_rbf", "rf"]
OS_SCORES = ["v3_auto_pooled", "v2_pooled", "knn_dist", "gmm_loglik", "mahalanobis",
             "svm_maxprob", "rf_maxprob"]
MAP_SIZES = (3, 5, 8, 12)
AUTO_SCORINGS = ("density", "joint_sorted")


def auto_model(seed):
    return KohonenDA3Auto(map_sizes=MAP_SIZES, scorings=AUTO_SCORINGS, trim="auto",
                          random_state=seed)


def prepare(dataset, seed, known_mask=None):
    X, y = L.load(dataset)
    if known_mask is not None:
        Xu = X[~known_mask(y)]
        X, y = X[known_mask(y)], y[known_mask(y)]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.32, stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    out = [sc.transform(Xtr), sc.transform(Xte), ytr, yte]
    if known_mask is not None:
        out.append(sc.transform(Xu))
    return out


def run_clean(dataset, seed):
    warnings.filterwarnings("ignore")
    Xtr, Xte, ytr, yte = prepare(dataset, seed)
    rows = []

    def timed(name, make, choice=lambda m: ""):
        t0 = time.perf_counter()
        m = make()
        t = time.perf_counter() - t0
        rows.append(dict(dataset=dataset, seed=seed, method=name,
                         acc=float((m.predict(Xte) == yte).mean()), fit_s=t, choice=choice(m)))
        return m

    timed("v1", lambda: KohonenDA(random_state=seed).fit(Xtr, ytr))
    timed("v2_joint", lambda: KohonenDA2(scoring="joint", random_state=seed).fit(Xtr, ytr))
    timed("v3_density_3x3", lambda: KohonenDA3(random_state=seed).fit(Xtr, ytr))
    timed("v3_auto_trim", lambda: auto_model(seed).fit(Xtr, ytr),
          lambda m: f"{m.map_size_}x{m.map_size_}/{m.scoring_}")
    base = {
        "lda": LinearDiscriminantAnalysis, "qda": lambda: QuadraticDiscriminantAnalysis(reg_param=1e-2),
        "gnb": GaussianNB, "gmm3": lambda: GMMClassifier(3, random_state=seed),
        "knn5": lambda: KNeighborsClassifier(5), "svm_rbf": SVC,
        "rf": lambda: RandomForestClassifier(200, random_state=seed),
    }
    for name, make in base.items():
        timed(name, lambda make=make: make().fit(Xtr, ytr))
    return rows


def run_openset(dataset, seed, alpha=0.05):
    warnings.filterwarnings("ignore")
    _, y_all = L.load(dataset)
    classes = np.unique(y_all)
    rng = np.random.default_rng(seed)
    unknown = rng.choice(classes, max(1, int(round(0.3 * len(classes)))), replace=False)
    known = lambda y: ~np.isin(y, unknown)  # noqa: E731
    Xtr, Xte, ytr, yte, Xu = prepare(dataset, seed, known)
    Xall = np.vstack([Xte, Xu])
    nk = len(Xte)

    s = {}
    au = auto_model(seed).fit(Xtr, ytr)
    pn = au.novelty_pvalues(Xall)
    s["v3_auto_pooled"] = -pn
    s["v2_pooled"] = -KohonenDA2(random_state=seed).fit(Xtr, ytr).novelty_pvalues(Xall)
    s["knn_dist"] = NearestNeighbors(n_neighbors=5).fit(Xtr).kneighbors(Xall)[0].mean(1)
    s["gmm_loglik"] = -GMMClassifier(3, random_state=seed).fit(Xtr, ytr) \
        ._joint_log_likelihood(Xall).max(1)
    s["mahalanobis"] = mahalanobis_score(Xtr, ytr, Xall)
    s["svm_maxprob"] = -SVC(probability=True, random_state=seed).fit(Xtr, ytr) \
        .predict_proba(Xall).max(1)
    s["rf_maxprob"] = -RandomForestClassifier(200, random_state=seed).fit(Xtr, ytr) \
        .predict_proba(Xall).max(1)

    lab = np.r_[np.zeros(nk), np.ones(len(Xu))]
    rows = [dict(dataset=dataset, seed=seed, score=k, auroc=roc_auc_score(lab, v),
                 det5=det_at(v[:nk], v[nk:])) for k, v in s.items()]
    rej = pn < alpha
    rows.append(dict(dataset=dataset, seed=seed, score="apriori_v3_auto",
                     frr=float(rej[:nk].mean()), det=float(rej[nk:].mean()),
                     choice=f"{au.map_size_}x{au.map_size_}/{au.scoring_}"))
    return rows


def summarise_clean(df, ref="v3_auto_trim"):
    acc = df.pivot_table(index="dataset", columns="method", values="acc")
    acc = acc.loc[[d for d in DATASETS[::-1] if d in acc.index], METHODS]
    print("Mean accuracy")
    print(acc.round(3).to_string())
    print("\nMean over datasets:")
    print(acc.mean().sort_values(ascending=False).round(4).to_string())
    print("\nAverage rank (1 = best):")
    print(acc.rank(axis=1, ascending=False).mean().sort_values().round(2).to_string())
    print(f"\n{ref} vs each method: paired t-test over splits per dataset (p<0.05; 5 splits)")
    rows = []
    for m in METHODS:
        if m == ref:
            continue
        b = w = 0
        for d in acc.index:
            sub = df[df.dataset == d].pivot_table(index="seed", columns="method", values="acc")
            diff = sub[ref] - sub[m]
            if diff.std() > 0 and stats.ttest_1samp(diff, 0).pvalue < 0.05:
                b += diff.mean() > 0
                w += diff.mean() < 0
        rows.append(dict(method=m, ref_better=b, ref_worse=w, ties=len(acc.index) - b - w))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nMean fit time per split, s:")
    t = df.pivot_table(index="dataset", columns="method", values="fit_s")
    print(t.loc[acc.index, METHODS].round(1).to_string())
    ch = df[df.method == ref].groupby("dataset").choice.agg(
        lambda c: ", ".join(f"{k} x{v}" for k, v in c.value_counts().items()))
    print(f"\n{ref} choices:")
    print(ch.loc[acc.index].to_string())


def summarise_openset(df):
    main = df[~df.score.str.startswith("apriori")]
    au = main.pivot_table(index="dataset", columns="score", values="auroc")
    au = au.loc[[d for d in DATASETS[::-1] if d in au.index], OS_SCORES]
    print("AUROC (unseen vs known)")
    print(au.round(3).to_string())
    print("\nAverage rank by AUROC:")
    print(au.rank(axis=1, ascending=False).mean().sort_values().round(2).to_string())
    det = main.pivot_table(index="dataset", columns="score", values="det5").loc[au.index, OS_SCORES]
    print("\nDetection at 5% false rejects (oracle threshold)")
    print(det.round(3).to_string())
    ap = df[df.score == "apriori_v3_auto"].groupby("dataset")[["frr", "det"]].mean()
    print("\nv3 auto, alpha=0.05 fixed in advance: false-reject / detection")
    print(ap.loc[au.index].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["clean", "openset", "report"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in DATASETS:  # fetch/cache once in the parent process
        L.load(d)
    t0 = time.time()
    tasks = [(d, s) for d in DATASETS for s in range(a.runs)]
    if a.experiment in ("clean", "openset"):
        fn = run_clean if a.experiment == "clean" else run_openset
        out = Parallel(n_jobs=a.jobs)(delayed(fn)(*t) for t in tasks)
        pd.DataFrame([r for rows in out for r in rows]).to_csv(OUT / f"{a.experiment}.csv",
                                                               index=False)
    if a.experiment in ("clean", "report") and (OUT / "clean.csv").exists():
        summarise_clean(pd.read_csv(OUT / "clean.csv"))
    if a.experiment in ("openset", "report") and (OUT / "openset.csv").exists():
        summarise_openset(pd.read_csv(OUT / "openset.csv"))
    print(f"\n[{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
