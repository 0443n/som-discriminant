"""Open-set recognition benchmark: can a classifier say "none of the known classes"?

Protocol: for every dataset with >= 3 classes and every class u, u is "unseen".
Train on the remaining classes (stratified 68/32 split of the known objects), then
score held-out known objects and *all* objects of u. Higher score = more "unknown".

Scores
  v2_conformal   : -max_c p_c(x), class-wise p-values (KohonenDA2, reject_mode="classwise")
  v2_pooled      : -novelty p-value of min_c d_c(x)   (KohonenDA2, reject_mode="pooled")
  v3_pooled_*    : -novelty p-value, neuron-mixture likelihood (KohonenDA3, diag/spherical)
  v3_classwise_diag : -max_c p_c(x), class-wise p-values on -log p(x|c) (KohonenDA3)
  v3_auto_pooled : as v3_pooled_diag, map size chosen out-of-fold (KohonenDA3Auto)
  nearest_map    : min_c d_c(x), raw BMU distance   (ablation: no conformal normalisation)
  knn_dist       : mean distance to the 5 nearest training objects
  gmm_loglik     : -max_c log p(x | c), per-class 3-component GMM
  mahalanobis    : min_c Mahalanobis distance to class means (shared covariance)
  lda_maxprob, svm_maxprob, rf_maxprob : -max posterior probability

Metrics
  auroc          : unseen vs. known, threshold-free
  det@5          : detection rate of unseen objects at the threshold that rejects 5% of
                   known test objects (oracle threshold, chosen on the test set)
  a-priori       : false-reject and detection rates of both v2 reject modes with
                   alpha = 0.05 fixed in advance (no oracle); the conformal guarantee
                   says false-reject <= ~5%

Usage: python scripts/openset.py [--runs 20]
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from kohonen_da import KohonenDA2, KohonenDA3, KohonenDA3Auto
from kohonen_da.baselines import GMMClassifier
from kohonen_da.datasets import ALL

OUT = Path(__file__).resolve().parent.parent / "results" / "benchmark"
DATASETS = ["iris", "wine", "digits", "multimodal", "rings", "heavy_tailed"]
SCORES = ["v2_conformal", "v2_pooled", "v3_pooled_diag", "v3_pooled_sph", "v3_classwise_diag",
          "v3_auto_pooled",
          "nearest_map", "knn_dist", "gmm_loglik", "mahalanobis",
          "lda_maxprob", "svm_maxprob", "rf_maxprob"]
ALPHA = 0.05


def mahalanobis_score(Xtr, ytr, X):
    classes = np.unique(ytr)
    mu = np.array([Xtr[ytr == c].mean(0) for c in classes])
    resid = Xtr - mu[np.searchsorted(classes, ytr)]
    cov = np.cov(resid, rowvar=False) + 1e-6 * np.eye(X.shape[1])
    P = np.linalg.pinv(cov)
    d = np.stack([np.einsum("ij,jk,ik->i", X - m, P, X - m) for m in mu], 1)
    return d.min(1)


def det_at(s_known, s_unknown, frr=0.05):
    thr = np.quantile(s_known, 1 - frr)
    return float((s_unknown > thr).mean())


def run_one(dataset, unknown, seed):
    warnings.filterwarnings("ignore")
    X, y = ALL[dataset](seed)
    kn = y != unknown
    Xtr, Xte, ytr, yte = train_test_split(X[kn], y[kn], test_size=0.32, stratify=y[kn],
                                          random_state=seed)
    Xu = X[~kn]
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte, Xu = sc.transform(Xtr), sc.transform(Xte), sc.transform(Xu)
    Xall = np.vstack([Xte, Xu])
    nk = len(Xte)

    s = {}
    v2 = KohonenDA2(random_state=seed).fit(Xtr, ytr)
    P = v2.pvalues(Xall)
    s["v2_conformal"] = -P.max(1)
    pn = v2.novelty_pvalues(Xall)
    s["v2_pooled"] = -pn
    v3 = KohonenDA3(covariance="diag", random_state=seed).fit(Xtr, ytr)
    pn3 = v3.novelty_pvalues(Xall)
    s["v3_pooled_diag"] = -pn3
    s["v3_classwise_diag"] = -v3.pvalues(Xall).max(1)
    v3.set_params(covariance="spherical")
    s["v3_pooled_sph"] = -v3.novelty_pvalues(Xall)
    pna = KohonenDA3Auto(random_state=seed).fit(Xtr, ytr).novelty_pvalues(Xall)
    s["v3_auto_pooled"] = -pna
    s["nearest_map"] = v2.bmu_distances(Xall).min(1)
    s["knn_dist"] = NearestNeighbors(n_neighbors=5).fit(Xtr).kneighbors(Xall)[0].mean(1)
    s["gmm_loglik"] = -GMMClassifier(3, random_state=seed).fit(Xtr, ytr) \
        ._joint_log_likelihood(Xall).max(1)
    s["mahalanobis"] = mahalanobis_score(Xtr, ytr, Xall)
    s["lda_maxprob"] = -LinearDiscriminantAnalysis().fit(Xtr, ytr).predict_proba(Xall).max(1)
    s["svm_maxprob"] = -SVC(probability=True, random_state=seed).fit(Xtr, ytr) \
        .predict_proba(Xall).max(1)
    s["rf_maxprob"] = -RandomForestClassifier(200, random_state=seed).fit(Xtr, ytr) \
        .predict_proba(Xall).max(1)

    lab = np.r_[np.zeros(nk), np.ones(len(Xu))]
    rows = []
    for name, sc_ in s.items():
        rows.append(dict(dataset=dataset, unknown=unknown, seed=seed, score=name,
                         auroc=roc_auc_score(lab, sc_),
                         det5=det_at(sc_[:nk], sc_[nk:])))
    # a-priori conformal operating point (no oracle)
    for name, rej in [("apriori_classwise", P.max(1) < ALPHA), ("apriori_pooled", pn < ALPHA),
                      ("apriori_v3_pooled_diag", pn3 < ALPHA), ("apriori_v3_auto", pna < ALPHA)]:
        rows.append(dict(dataset=dataset, unknown=unknown, seed=seed, score=name,
                         frr=float(rej[:nk].mean()), det=float(rej[nk:].mean())))
    return rows


def summarise(df):
    main = df[~df.score.str.startswith("apriori")]
    # average over unseen classes and seeds within a dataset
    auroc = main.pivot_table(index="dataset", columns="score", values="auroc")[SCORES]
    auroc = auroc.loc[[d for d in DATASETS if d in auroc.index]]
    det = main.pivot_table(index="dataset", columns="score", values="det5")[SCORES]
    det = det.loc[auroc.index]
    print("AUROC (unseen vs known), mean over held-out classes x splits")
    print(auroc.round(3).to_string())
    print("\nAverage rank by AUROC (1 = best):")
    print(auroc.rank(axis=1, ascending=False).mean().sort_values().round(2).to_string())
    print("\nDetection rate at 5% false rejects (oracle threshold)")
    print(det.round(3).to_string())
    print("\nAverage rank by det@5:")
    print(det.rank(axis=1, ascending=False).mean().sort_values().round(2).to_string())

    ref = "v3_pooled_diag"
    print(f"\n{ref} vs each score: paired Wilcoxon on AUROC over (class, split), p<0.05")
    rows = []
    for m in [x for x in SCORES if x != ref]:
        b = w = 0
        for d in auroc.index:
            sub = main[main.dataset == d].pivot_table(index=["unknown", "seed"],
                                                       columns="score", values="auroc")
            diff = sub[ref] - sub[m]
            if (diff != 0).any() and stats.wilcoxon(diff, zero_method="zsplit").pvalue < 0.05:
                b += diff.mean() > 0
                w += diff.mean() < 0
        rows.append(dict(score=m, v2_better=b, v2_worse=w, ties=len(auroc.index) - b - w))
    print(pd.DataFrame(rows).to_string(index=False))

    ap = df[df.score.str.startswith("apriori")].pivot_table(
        index="dataset", columns="score", values=["frr", "det"])
    print(f"\nv2 with alpha={ALPHA} fixed in advance (no oracle): false-reject / detection")
    print(ap.loc[auroc.index].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--report", action="store_true", help="only summarise saved CSV")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "openset.csv"
    t0 = time.time()
    if not a.report:
        tasks = [(d, u, s) for d in DATASETS for u in np.unique(ALL[d](0)[1])
                 for s in range(a.runs)]
        out = Parallel(n_jobs=a.jobs)(delayed(run_one)(*t) for t in tasks)
        pd.DataFrame([r for rows in out for r in rows]).to_csv(path, index=False)
    summarise(pd.read_csv(path))
    print(f"\n[{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
