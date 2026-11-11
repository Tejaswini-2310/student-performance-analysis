"""
Predicting student pass/fail at different points: background only (A), + G1 (B), + G1 and G2 (C).
 
Setup:
  1. Download the UCI "Student Performance" dataset (student.zip) and put student-mat.csv and
     student-por.csv inside a folder called data/ next to this script. The files use ';' as separator.
  2. pip install scikit-learn pandas numpy matplotlib scipy
  3. python student_pass_fail_experiment.py
 
Target: fail = 1 if final grade G3 < 10, else 0 (pass). "Fail" is the positive class, so recall = share of
failing students that were caught. All numbers in your paper must come from the files written to ./results/.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.*",
    category=UserWarning
)
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import clone
 
warnings.filterwarnings("ignore")
 
FILES = {"Math": "data/student-mat.csv", "Portuguese": "data/student-por.csv"}
N_REPEATS, N_FOLDS, SEED, OUT = 10, 5, 42, "results"
SCENARIOS = {"A: background only": ["G1", "G2", "G3"],   # columns to drop
             "B: + G1": ["G2", "G3"],
             "C: + G1 and G2": ["G3"]}
os.makedirs(OUT, exist_ok=True)
 
 
def load(path):
    df = pd.read_csv(path, sep=";")
    y = (df["G3"] < 10).astype(int)          # 1 = fail
    return df, y
 
 
def make_models(X):
    cat = X.select_dtypes(include="object").columns.tolist()
    num = [c for c in X.columns if c not in cat]
 
    def pre(scale):
        return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                                  ("num", StandardScaler() if scale else "passthrough", num)])
    return {
        "LogReg": Pipeline([("p", pre(True)), ("m", LogisticRegression(max_iter=2000))]),
        "DecisionTree": Pipeline([("p", pre(False)), ("m", DecisionTreeClassifier(max_depth=5, random_state=SEED))]),
        "RandomForest": Pipeline([("p", pre(False)), ("m", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1))]),
    }
 
 
def run():
    rows, imp_rows = [], []
    for dname, path in FILES.items():
        df, y = load(path)
        print(f"\n=== {dname}: {len(df)} students, fail rate {y.mean():.3f}")
        for sname, drop in SCENARIOS.items():
            X = df.drop(columns=drop)
            models = make_models(X)
            cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
            for mname, pipe in models.items():
                res = cross_validate(pipe, X, y, cv=cv, n_jobs=1,
                                     scoring={"accuracy": "accuracy", "f1": "f1", "recall": "recall", "auc": "roc_auc"})
                for k in range(len(res["test_accuracy"])):
                    rows.append(dict(dataset=dname, scenario=sname, model=mname, fold=k,
                                     accuracy=res["test_accuracy"][k], f1=res["test_f1"][k],
                                     recall=res["test_recall"][k], auc=res["test_auc"][k]))
            # permutation importance, scenario A only, Random Forest, one ranking per repeat
            if sname.startswith("A"):
                for r in range(N_REPEATS):
                    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + r)
                    imp = np.zeros(X.shape[1])
                    for tr, te in skf.split(X, y):
                        m = clone(models["RandomForest"]).fit(X.iloc[tr], y.iloc[tr])
                        pi = permutation_importance(m, X.iloc[te], y.iloc[te], n_repeats=5,
                                                    random_state=SEED + r, scoring="roc_auc")
                        imp += pi.importances_mean
                    imp /= N_FOLDS
                    ranks = pd.Series(imp, index=X.columns).rank(ascending=False)
                    for f, v in ranks.items():
                        imp_rows.append(dict(dataset=dname, repeat=r, feature=f, rank=v))
        print("  done")
    raw = pd.DataFrame(rows); raw.to_csv(f"{OUT}/raw_scores.csv", index=False)
    imp = pd.DataFrame(imp_rows); imp.to_csv(f"{OUT}/permutation_ranks_raw.csv", index=False)
    return raw, imp
 
 
def analyse(raw, imp):
    g = raw.groupby(["dataset", "scenario", "model"])[["accuracy", "f1", "recall", "auc"]]
    (g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)).to_csv(f"{OUT}/summary_mean_std.csv")
    means = g.mean()
    means.round(4).to_csv(f"{OUT}/summary_means.csv")
 
    # change in recall and accuracy between scenarios (A -> C)
    a = means.xs("A: background only", level="scenario")
    c = means.xs("C: + G1 and G2", level="scenario")
    (c - a).round(4).to_csv(f"{OUT}/change_A_to_C.csv")
 
    # permutation-importance stability (scenario A, Random Forest)
    st = []
    for d, sub in imp.groupby("dataset"):
        piv = sub.pivot(index="feature", columns="repeat", values="rank")
        rho = [spearmanr(piv[i], piv[j])[0] for i in piv.columns for j in piv.columns if i < j]
        summ = pd.DataFrame({"mean_rank": piv.mean(axis=1), "std_rank": piv.std(axis=1)}).sort_values("mean_rank")
        summ.round(2).to_csv(f"{OUT}/permutation_rank_summary_{d}.csv")
        st.append(dict(dataset=d, mean_pairwise_spearman=np.mean(rho)))
        # figure: top 10 factors
        top = summ.head(10)[::-1]
        plt.figure(figsize=(5, 3.6))
        plt.barh(top.index, top["mean_rank"], xerr=top["std_rank"])
        plt.xlabel("Mean rank across 10 repeats (1 = most important)"); plt.title(f"{d}: top factors, background only")
        plt.tight_layout(); plt.savefig(f"{OUT}/fig2_top_factors_{d}.png", dpi=200); plt.close()
    pd.DataFrame(st).round(3).to_csv(f"{OUT}/permutation_stability.csv", index=False)
 
    # Figure 1: mean F1 (fail class) by scenario, per dataset and model
    ds = list(raw.dataset.unique()); sc = list(SCENARIOS); ms = list(raw.model.unique())
    fig, axes = plt.subplots(1, len(ds), figsize=(5.5 * len(ds), 3.8), sharey=True)
    axes = np.atleast_1d(axes); x = np.arange(len(ms))
    for ax, d in zip(axes, ds):
        for i, s in enumerate(sc):
            vals = [means.loc[(d, s, m), "f1"] for m in ms]
            ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=s)
        ax.set_title(d); ax.set_xticks(x); ax.set_xticklabels(ms)
    axes[0].set_ylabel("Mean F1 (fail class)"); axes[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig1_f1_by_scenario.png", dpi=200); plt.close()
 
    # Figure 3: recall for failing students, box plot by scenario (all models and datasets pooled)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.boxplot([raw[raw.scenario == s].recall.values for s in sc])
    ax.set_xticklabels([s.split(":")[0] for s in sc])
    ax.set_ylabel("Recall for failing students"); ax.set_xlabel("Scenario")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig3_recall_by_scenario.png", dpi=200); plt.close()
    print("Saved tables and figures to ./results/")
 
 
if __name__ == "__main__":
    analyse(*run())