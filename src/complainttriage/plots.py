"""Figures drawn from results/results.json (no model needed)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main(results=ROOT / "results" / "results.json", out=ROOT / "results" / "summary.png"):
    r = json.loads(Path(results).read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    dec = r["test"]["calibration_deciles"]
    pred = [d["mean_predicted"] * 100 for d in dec]
    obs = [d["observed"] * 100 for d in dec]
    ax[0].plot(pred, obs, marker="o", color="#2a6f97")
    lim = max(pred + obs) * 1.08
    ax[0].plot([0, lim], [0, lim], ls="--", color="#999")
    ax[0].set(xlabel="Predicted chance of monetary relief (%)", ylabel="Observed relief rate (%)",
              title="Calibration by decile, 2025-2026 test set")

    comp = r["test_auroc_by_company"]
    names = [c.split(",")[0].replace(" FINANCIAL CORPORATION", "").replace(" & COMPANY", "").title()[:26] for c in comp]
    vals = [v["auroc"] for v in comp.values()]
    ax[1].barh(names[::-1], vals[::-1], color=["#c0392b" if v < 0.6 else "#2a6f97" for v in vals[::-1]])
    ax[1].axvline(0.5, ls="--", color="#999")
    ax[1].axvline(r["test"]["auroc"], ls=":", color="#333", label=f"All complaints ({r['test']['auroc']:.3f})")
    ax[1].set(xlim=(0.3, 0.9), xlabel="Test AUROC", title="Ranking quality for the 6 largest banks")
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
