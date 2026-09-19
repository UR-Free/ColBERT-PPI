"""Recompute manuscript measurements from prepared predictions and labels.

These functions do not load encoders, select checkpoints or train models.
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(y, s):
    return {
        "auprc": float(average_precision_score(y, s)),
        "auroc": float(roc_auc_score(y, s)),
    }


def check_retrieval(source_data):
    """Return PINDER comparator and PINDER/Y2H ablation measurements."""
    out = []
    # Fig 2a: score-level replication of every external and full-model labelled benchmark result.
    scores = pd.read_csv(source_data / "Fig2_PPI_Retrieval_fixed_1to100_scores.csv")
    exp = pd.read_csv(
        source_data / "Fig2_PPI_Retrieval_fixed_1to100_metrics.csv"
    ).set_index("method")
    for name, g in scores.groupby("method"):
        m = metrics(g.label, g.score)
        for k, v in m.items():
            assert np.isclose(v, exp.loc[name, "test_" + k], atol=1e-10), (name, k, v)
        out.append({"figure": "Fig2a", "model": name, **m})
    print("Fig2a score-level checks passed", flush=True)
    # Fig4: full/sequence/single on the common frozen PINDER and Y2H labels.
    for task, fn in [
        ("PINDER", "Fig4_three_arm_PINDER_pair_scores_and_strata.csv"),
        ("Y2H", "Fig4_three_arm_Y2H_pair_scores_and_strata.csv"),
    ]:
        t = pd.read_csv(source_data / fn)
        for name in [
            "Classic ColBERT-PPI",
            "Sequence-only multi-vector",
            "AA+3Di single-vector",
        ]:
            out.append(
                {
                    "figure": "Fig4 retrieval full labelled set",
                    "task": task,
                    "model": name,
                    **metrics(t.label, t[name]),
                }
            )

    return out


def check_localisation(root, output_dir):
    """Check chain and contact metrics for each complex, then average complexes."""
    source_data = root / "data/source_data"
    # Fig3 / Fig4 localisation: recompute each chain then give each complex equal weight.
    expected = pd.read_csv(source_data / "Fig4_three_arm_localisation_per_complex.csv")
    loc = []
    for arm, method in [
        ("full", "Classic ColBERT-PPI"),
        ("sequence", "Sequence-only multi-vector"),
    ]:
        data = np.load(root / f"data/localisation/{arm}.npz")
        for row in expected[expected.method.eq(method)].itertuples():
            i = int(row.pair_index)
            a = metrics(data[f"{i}_a_label"], data[f"{i}_a_score"])
            b = metrics(data[f"{i}_b_label"], data[f"{i}_b_score"])
            c = metrics(data[f"{i}_contact_label"], data[f"{i}_contact_score"])
            result = {"model": method, "pair_index": i}
            for k in ["auroc", "auprc"]:
                v = (a[k] + b[k]) / 2
                assert np.isclose(v, getattr(row, "interface_" + k), atol=1e-9), (
                    arm,
                    i,
                    k,
                    v,
                )
                assert np.isclose(c[k], getattr(row, "contact_" + k), atol=1e-9), (
                    arm,
                    i,
                    k,
                    c[k],
                )
                result["interface_" + k] = v
                result["contact_" + k] = c[k]
            loc.append(result)
        print(arm, "226 localisation checks passed", flush=True)
    pd.DataFrame(loc).to_csv(output_dir / "localisation_recomputed.csv", index=False)

    return (
        pd.DataFrame(loc)
        .groupby("model")
        .mean(numeric_only=True)
        .drop(columns=["pair_index"])
        .to_dict("index")
    )


def check_protein_rna(root, output_dir):
    """Check all 19 runs across four cohorts and summarise common-cohort gains."""
    source_data = root / "data/source_data"
    # Fig5: all 19 internal/external runs, four independently labelled cohorts.
    labels = np.load(root / "data/pri/labels.npz")
    cohorts = [k[:-9] for k in labels.files if k.endswith("_positive")]
    pri = []
    for meta in json.loads((root / "data/pri/methods.json").read_text()):
        s = np.load(root / "data/pri" / meta["file"])["scores"]
        for cohort in cohorts:
            p = labels[cohort + "_positive"]
            n = labels[cohort + "_negative"]
            assert not (p & n).any()
            mask = p | n
            pri.append(
                {k: meta[k] for k in ["run_id", "representation", "fraction", "seed"]}
                | {
                    "cohort": cohort,
                    "positives": int(p.sum()),
                    "negatives": int(n.sum()),
                    **metrics(p[mask], s[mask]),
                }
            )
    p = pd.DataFrame(pri)
    expected = pd.read_csv(source_data / "Fig5_all_methods_run_values.csv")
    for row in p.itertuples():
        e = expected[expected.run_id.eq(row.run_id) & expected.cohort.eq(row.cohort)]
        assert len(e) == 1
        for key in ["auprc", "auroc"]:
            assert np.isclose(getattr(row, key), e.iloc[0][key], atol=1e-10), (
                row.run_id,
                key,
            )
    p.to_csv(output_dir / "pri_recomputed.csv", index=False)
    clean = p[
        p.cohort.eq("common_exact_source_clean") & p.representation.ne("external")
    ]
    g = clean.groupby(["representation", "fraction"]).auprc.mean()
    m0 = g.loc["multi_vector", 0]
    m1 = g.loc["multi_vector", 100]
    s1 = g.loc["single_vector", 100]

    return len(p), {
        "multi_0": m0,
        "multi_100": m1,
        "single_100": s1,
        "ppi_gain_percent": 100 * (m1 / m0 - 1),
        "multi_vs_single_gain_percent": 100 * (m1 / s1 - 1),
    }
