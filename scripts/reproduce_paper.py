"""Reproduce Section 4 of Komarov & Kuravsky (2023) on Fisher's iris data.

Setup inferred from the paper:
  * 3 maps of size 3x3 (9 neurons each), raw (unscaled) features — the BMU distance
    ranges match Fig. 3 only without scaling;
  * 16 test objects per class (confusion matrix sums to 48) -> 34 train per class;
  * histograms are built from the test sample, and the confusion matrix is computed on
    the same test sample (literal reading of the paper — see "protocols" below).

Protocols compared over many random splits:
  paper    : SOMs on train, histograms on test, evaluate on test      (as in the paper; leaky)
  train-cal: SOMs on train, histograms on train, evaluate on test      (no leakage)
  3-way    : SOMs on train, histograms on separate calibration split, evaluate on test
  LDA      : Fisher LDA on train, evaluate on test (same splits)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from kohonen_da import KohonenDA

OUT = Path(__file__).resolve().parent.parent / "results"
NAMES = ["setosa", "versicolor", "virginica"]


def paper_split(X, y, seed):
    return train_test_split(X, y, test_size=48, stratify=y, random_state=seed)


def single_run(X, y, seed):
    Xtr, Xte, ytr, yte = paper_split(X, y, seed)
    m = KohonenDA(map_size=3, random_state=seed).fit(Xtr, ytr, Xte, yte)
    pred = m.predict(Xte)
    lda = LinearDiscriminantAnalysis().fit(Xtr, ytr)

    # paper layout: rows = predicted, columns = true
    cm = confusion_matrix(yte, pred).T
    cm_lda = confusion_matrix(yte, lda.predict(Xte)).T
    print(f"Single run (seed={seed}), rows=predicted, cols=true")
    print("KohonenDA:\n", cm, f"\naccuracy = {np.trace(cm)}/{cm.sum()}")
    print("LDA:\n", cm_lda, f"\naccuracy = {np.trace(cm_lda)}/{cm_lda.sum()}")
    print(f"undecided (all-zero probabilities): {m.n_undecided(Xte)}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for k, ax in enumerate(axes):
        f, e = m.hist_freqs_[k], m.hist_edges_[k]
        ax.bar(e[:-1], f, width=np.diff(e), align="edge", color="#3b75b0")
        ax.set_xlim(0, max(0.6, e[-1]))
        ax.set_ylim(0, 1)
        ax.set_title(f"{k} class ({NAMES[k]})")
        ax.set_xlabel("distance to BMU")
    axes[0].set_ylabel("normalised frequency")
    fig.suptitle("Reproduction of Fig. 3")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_reproduction.png", dpi=130)
    plt.close(fig)


def repeated(X, y, n_runs):
    rows = []
    for seed in range(n_runs):
        Xtr, Xte, ytr, yte = paper_split(X, y, seed)
        # 3-way: carve 16/class calibration set out of the 34/class train set
        Xtr3, Xcal, ytr3, ycal = train_test_split(
            Xtr, ytr, test_size=48, stratify=ytr, random_state=seed
        )

        m = KohonenDA(map_size=3, random_state=seed).fit(Xtr, ytr, Xte, yte)
        acc_paper = (m.predict(Xte) == yte).mean()
        und_paper = m.n_undecided(Xte)
        # own-class probability exactly 0 (the "too close to a prototype" problem shows
        # up as zeros below the first bin edge, too far -> above the last one)
        S = m.histogram_scores(Xte)
        own_zero = int((S[np.arange(len(yte)), yte] == 0).sum())

        m.calibrate(Xtr, ytr)
        acc_tc = (m.predict(Xte) == yte).mean()
        und_tc = m.n_undecided(Xte)
        S = m.histogram_scores(Xte)
        own_zero_tc = int((S[np.arange(len(yte)), yte] == 0).sum())

        m3 = KohonenDA(map_size=3, random_state=seed).fit(Xtr3, ytr3, Xcal, ycal)
        acc_3w = (m3.predict(Xte) == yte).mean()
        und_3w = m3.n_undecided(Xte)

        acc_lda = (LinearDiscriminantAnalysis().fit(Xtr, ytr).predict(Xte) == yte).mean()
        rows.append(
            dict(
                seed=seed,
                paper=acc_paper,
                train_cal=acc_tc,
                three_way=acc_3w,
                lda=acc_lda,
                undecided_paper=und_paper,
                undecided_train_cal=und_tc,
                undecided_three_way=und_3w,
                own_zero_paper=own_zero,
                own_zero_train_cal=own_zero_tc,
            )
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0, help="seed for the single illustrative run")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    X, y = load_iris(return_X_y=True)
    single_run(X, y, args.seed)

    df = repeated(X, y, args.runs)
    df.to_csv(OUT / "repeated_runs.csv", index=False)

    print(f"\n=== {args.runs} random 34/16-per-class splits ===")
    acc = df[["paper", "train_cal", "three_way", "lda"]]
    print("accuracy  mean / std / min / max")
    print(acc.agg(["mean", "std", "min", "max"]).T.round(4).to_string())
    print("\nerrors out of 48 (mean):")
    print(((1 - acc) * 48).mean().round(2).to_string())
    print("\nKohonenDA vs LDA on the same split (paper protocol):")
    d = df["paper"] - df["lda"]
    print(f"  better {(d > 0).mean():.1%}  equal {(d == 0).mean():.1%}  worse {(d < 0).mean():.1%}")
    d = df["train_cal"] - df["lda"]
    print("KohonenDA vs LDA (train-cal protocol, no leakage):")
    print(f"  better {(d > 0).mean():.1%}  equal {(d == 0).mean():.1%}  worse {(d < 0).mean():.1%}")
    print("\nper-split mean counts (out of 48 test objects):")
    print(
        df[[c for c in df.columns if c.startswith(("undecided", "own_zero"))]]
        .mean()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
