"""Evaluate KohonenDA v2 against the 2023 method and classical baselines.

A. Closed-set accuracy on the same random 34/16-per-class iris splits as
   reproduce_paper.py (test_size=48, identical seeds).
B. Coverage of the conformal p-values: P(p_true < alpha) on test objects.
C. Open-set recognition: train on two iris species, test on held-out objects of the
   known species plus *all* objects of the unseen species. v2 rejects at max p < alpha;
   the 2023 method's only "reject" is the all-zero-probability case.

Usage: python scripts/evaluate_v2.py [--runs 200] [--dataset iris|wine|breast_cancer] [--scale]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from kohonen_da import KohonenDA, KohonenDA2

OUT = Path(__file__).resolve().parent.parent / "results"
ALPHAS = (0.05, 0.10, 0.20)


def load(name):
    loader = {"iris": datasets.load_iris, "wine": datasets.load_wine,
              "breast_cancer": datasets.load_breast_cancer}[name]
    return loader(return_X_y=True)


def split(X, y, seed, scale):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.32, stratify=y, random_state=seed)
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    return Xtr, Xte, ytr, yte


def closed_set(X, y, runs, scale, map_size):
    rows, cover = [], []
    for seed in range(runs):
        Xtr, Xte, ytr, yte = split(X, y, seed, scale)
        r = dict(seed=seed)

        v1 = KohonenDA(map_size=map_size, random_state=seed).fit(Xtr, ytr, Xte, yte)
        r["v1_paper"] = (v1.predict(Xte) == yte).mean()
        r["nearest_map"] = (v1.classes_[v1.bmu_distances(Xte).argmin(1)] == yte).mean()
        v1.calibrate(Xtr, ytr)
        r["v1_train_cal"] = (v1.predict(Xte) == yte).mean()

        v2 = KohonenDA2(map_size=map_size, random_state=seed).fit(Xtr, ytr)
        r["v2_pvalue"] = (v2.predict(Xte) == yte).mean()
        v2.set_params(scoring="joint")
        r["v2_joint"] = (v2.predict(Xte) == yte).mean()

        for name, est in [("lda", LinearDiscriminantAnalysis()),
                          ("qda", QuadraticDiscriminantAnalysis(reg_param=1e-3)),
                          ("knn5", KNeighborsClassifier(5))]:
            r[name] = (est.fit(Xtr, ytr).predict(Xte) == yte).mean()
        rows.append(r)

        P = v2.pvalues(Xte)
        p_true = P[np.arange(len(yte)), np.searchsorted(v2.classes_, yte)]
        sets = [P >= a for a in ALPHAS]
        cover.append({**{f"miss@{a}": (p_true < a).mean() for a in ALPHAS},
                      **{f"setsize@{a}": s.sum(1).mean() for a, s in zip(ALPHAS, sets)}})
    return pd.DataFrame(rows), pd.DataFrame(cover)


def open_set(X, y, runs, scale, map_size, alpha=0.05):
    rows = []
    for unknown in np.unique(y):
        kn = y != unknown
        Xk, yk, Xu = X[kn], y[kn], X[~kn]
        for seed in range(runs):
            Xtr, Xte, ytr, yte = train_test_split(Xk, yk, test_size=0.32, stratify=yk,
                                                  random_state=seed)
            Xu_ = Xu
            if scale:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte, Xu_ = sc.transform(Xtr), sc.transform(Xte), sc.transform(Xu)

            v2 = KohonenDA2(map_size=map_size, reject_alpha=alpha, random_state=seed).fit(Xtr, ytr)
            pk, pu = v2.predict(Xte), v2.predict(Xu_)
            acc_k = (pk[pk != -1] == yte[pk != -1]).mean() if (pk != -1).any() else np.nan

            # threshold-free detection quality and a second operating point
            mk, mu = v2.pvalues(Xte).max(1), v2.pvalues(Xu_).max(1)
            auroc = roc_auc_score(np.r_[np.zeros(len(mk)), np.ones(len(mu))], -np.r_[mk, mu])

            v1 = KohonenDA(map_size=map_size, zero_fallback="none",
                           random_state=seed).fit(Xtr, ytr)  # train-cal, no leakage
            rows.append(dict(
                unknown=unknown, seed=seed,
                v2_false_reject=(pk == -1).mean(),
                v2_detect_unknown=(pu == -1).mean(),
                v2_acc_accepted=acc_k,
                v2_false_reject_a20=(mk < 0.2).mean(),
                v2_detect_unknown_a20=(mu < 0.2).mean(),
                v2_auroc=auroc,
                v1_false_reject=v1.n_undecided(Xte) / len(Xte),
                v1_detect_unknown=v1.n_undecided(Xu_) / len(Xu_),
            ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--open-runs", type=int, default=50)
    ap.add_argument("--dataset", default="iris")
    ap.add_argument("--scale", action="store_true", help="standardise features (fit on train)")
    ap.add_argument("--map-size", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tag = f"{a.dataset}{'_scaled' if a.scale else ''}_m{a.map_size}"
    X, y = load(a.dataset)

    acc, cov = closed_set(X, y, a.runs, a.scale, a.map_size)
    acc.to_csv(OUT / f"v2_closed_{tag}.csv", index=False)
    n_test = int(round(0.32 * len(y)))
    methods = [c for c in acc.columns if c != "seed"]
    print(f"=== A. closed set: {a.dataset}, {a.runs} splits, map {a.map_size}x{a.map_size}, "
          f"scale={a.scale}, test n={n_test} ===")
    summ = acc[methods].agg(["mean", "std"]).T
    summ["errors"] = (1 - summ["mean"]) * n_test
    summ["vs_lda_better"] = [(acc[m] > acc["lda"]).mean() for m in methods]
    summ["vs_lda_worse"] = [(acc[m] < acc["lda"]).mean() for m in methods]
    print(summ.round(4).to_string())

    print("\n=== B. conformal coverage (v2, out-of-sample) ===")
    print(cov.mean().round(4).to_string())

    if len(np.unique(y)) >= 3:
        op = open_set(X, y, a.open_runs, a.scale, a.map_size)
        op.to_csv(OUT / f"v2_openset_{tag}.csv", index=False)
        print(f"\n=== C. open set (alpha=0.05), {a.open_runs} splits per held-out class ===")
        print(op.drop(columns="seed").groupby("unknown").mean().round(3).T.to_string())


if __name__ == "__main__":
    main()
