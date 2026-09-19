"""Replot numerical evidence and export the supplied final figure compositions."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/figures"))
    p.add_argument(
        "--official",
        action="store_true",
        help="Export the final SVG compositions to PDF",
    )
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / "data/source_data"
    args.output.mkdir(parents=True, exist_ok=True)
    if args.official:
        import cairosvg

        for path in (root / "data/figure_templates").glob("*.svg"):
            cairosvg.svg2pdf(
                url=str(path), write_to=str(args.output / (path.stem + ".pdf"))
            )
        return
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    def save(fig, name):
        fig.savefig(args.output / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    t = pd.read_csv(data / "Fig2_precision_recall_curves.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, group in t.groupby("method"):
        ax.plot(group.recall, group.precision, label=name)
    ax.set(xlabel="Recall", ylabel="Precision", title="PINDER partner retrieval")
    ax.legend(frameon=False)
    save(fig, "Fig2_Retrieval_Replot")
    t = pd.read_csv(data / "Fig3_Contact_Interface_comparison_per_complex.csv")
    wide = t.pivot(index="pair_index", columns="method", values="interface_auroc")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, other in zip(axes, ["FlashPPI", "RaftPPI"]):
        ax.scatter(wide[other], wide["ColBERT-PPI"], s=9, color="#e36f1c", alpha=0.6)
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set(
            xlabel=other + " interface AUROC",
            ylabel="ColBERT-PPI interface AUROC",
            aspect="equal",
        )
    save(fig, "Fig3_Localisation_Replot")
    t = pd.read_csv(data / "Fig4_three_arm_localisation_per_complex.csv")
    means = t.groupby("method")[["interface_auroc", "contact_auroc"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    means.plot.bar(ax=ax, rot=15)
    ax.set(ylabel="Mean AUROC", xlabel="", ylim=(0, 1))
    save(fig, "Fig4_Ablation_Replot")
    t = pd.read_csv(data / "Fig5_common_exact_source_clean_comparison.csv")
    dose = pd.read_csv(data / "Fig5_PPI_data_dose_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].barh(t.method, t.auprc, color="#e36f1c")
    axes[0].set(xlabel="AUPRC", title="Protein–RNA retrieval")
    axes[1].errorbar(
        dose.fraction, dose.auprc_mean, yerr=dose.auprc_sd, marker="o", capsize=3
    )
    axes[1].set(
        xlabel="PPI training data (%)", ylabel="AUPRC", title="PPI initialization"
    )
    save(fig, "Fig5_Transfer_Replot")


if __name__ == "__main__":
    main()
