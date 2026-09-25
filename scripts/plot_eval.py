"""Figures for scripts/eval.py: `uv run python scripts/eval.py plot`."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

MODEL_NAMES = {"laya": "Laya", "gliner:fp32": "GLiNER fp32", "gliner:int8": "GLiNER int8", "gliner:fp16": "GLiNER fp16",
               "gliner:fp32:orders=2": "GLiNER fp32 · 2 orders", "gliner:fp32:orders=3": "GLiNER fp32 · 3 orders",
               "gliner:int8:orders=3": "GLiNER int8 · 3 orders"}
PALETTE = {"Laya": "#2a9d8f", "GLiNER fp32": "#e9c46a", "GLiNER int8": "#f4a261", "GLiNER fp16": "#9c6644",
           "GLiNER fp32 · 2 orders": "#b5179e", "GLiNER fp32 · 3 orders": "#e76f51", "GLiNER int8 · 3 orders": "#6a4c93"}
MARKERS = {"noul": "o", "choice": "s", "score": "D", "any": "^", "all": "v", "votes=2": "P",
           "noul|choice": "X", "noul&choice": "*"}
INK, MUTED = "#264653", "#8d99ae"


def roc(scored: list[tuple[float, bool]]) -> tuple[list[float], list[float]]:
    pos = sum(y for _, y in scored)
    neg = len(scored) - pos
    fpr, tpr, tp, fp = [0.0], [0.0], 0, 0
    for s, y in sorted(scored, key=lambda x: -x[0]):
        tp, fp = tp + y, fp + (not y)
        fpr.append(fp / neg)
        tpr.append(tp / pos)
    return fpr, tpr


def frame(results: list[dict], test: list[dict]) -> pd.DataFrame:
    n_ben = sum(r["kind"] == "benign" for r in test)
    n_hard = sum(r["kind"] == "hard_negative" for r in test)
    out = []
    for r in results:
        m = r["tuned"]
        out.append({
            "model": MODEL_NAMES.get(r["model"], r["model"]), "config": r["config"], "recall": m["recall"],
            "fp": (m["benign_fp"] * n_ben + m["hard_fp"] * n_hard) / (n_ben + n_hard),
            "auc": m["auc"], "twins": m["pairs"][0], "n_twins": m["pairs"][1], "latency": m["latency"],
        })
    return pd.DataFrame(out)


def overview(results: list[dict], test: list[dict], out: Path, set_name: str) -> Path:
    df = frame(results, test)
    models = [m for m in PALETTE if m in set(df["model"])]
    configs = [c for c in MARKERS if c in set(df["config"])]
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))
    (ax_op, ax_roc), (ax_twin, ax_lat) = axes

    # A. operating points: recall vs false-positive rate, at each config's dev-tuned threshold
    sns.scatterplot(data=df, x="fp", y="recall", hue="model", style="config", palette=PALETTE, markers=MARKERS,
                    hue_order=models, style_order=configs, s=150, edgecolor="white", linewidth=0.8, ax=ax_op,
                    legend=False, zorder=3)
    # label each model's best trade-off (highest recall - FP), pushed clear of the cluster
    offsets = [(-95, 38), (40, 34), (30, -34), (-60, -40)]
    labeled: set[tuple[float, float]] = set()
    for model, offset in zip(models, offsets):
        best = df[df.model == model].assign(gap=lambda d: d.recall - d.fp).nlargest(1, "gap").iloc[0]
        if (best.fp, best.recall) in labeled:  # e.g. score is the same point with or without noul_orders
            continue
        labeled.add((best.fp, best.recall))
        ax_op.annotate(f"{model}: {best.config}", (best.fp, best.recall), xytext=offset, textcoords="offset points",
                       fontsize=9.5, color=PALETTE[model], fontweight="bold",
                       arrowprops={"arrowstyle": "-", "color": PALETTE[model], "linewidth": 1})
    ax_op.set(xlim=(0, min(1.0, df.fp.max() + 0.08)), ylim=(max(0.0, df.recall.min() - 0.08), min(1.0, df.recall.max() + 0.08)),
              xlabel="false-positive rate (benign + hard negatives)",
              ylabel="harmful recall",
              title="A · Operating points (" + ("dev-tuned threshold)" if set_name == "core" else "core-dev threshold)"))
    ax_op.text(0.02, 0.98, "better: up and to the left", fontsize=9, color=MUTED, va="top", transform=ax_op.transAxes)

    # B. ROC of each model's best config by AUC, the rest faint
    for r in results:
        name = MODEL_NAMES.get(r["model"], r["model"])
        fpr, tpr = roc(r["test_scores"])
        ax_roc.plot(fpr, tpr, color=PALETTE[name], alpha=0.12, linewidth=1)
    for model in models:
        best = max((r for r in results if MODEL_NAMES.get(r["model"]) == model), key=lambda r: r["tuned"]["auc"])
        fpr, tpr = roc(best["test_scores"])
        ax_roc.plot(fpr, tpr, color=PALETTE[model], linewidth=2.6,
                    label=f"{model} · {best['config']} (AUC {best['tuned']['auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1)
    ax_roc.set(xlim=(0, 1), ylim=(0, 1), xlabel="false-positive rate", ylabel="true-positive rate",
               title="B · ROC, best config per model (others faint)")
    ax_roc.legend(loc="lower right", frameon=True, fontsize=9)

    # C. twins told apart: harmful twin flagged and its benign twin passed
    sns.barplot(data=df, x="config", y="twins", hue="model", palette=PALETTE, hue_order=models, order=configs,
                ax=ax_twin, edgecolor="white", legend=False)
    n_twins = int(df["n_twins"].max())
    ax_twin.axhline(n_twins, color=MUTED, linestyle=":", linewidth=1)
    ax_twin.text(len(configs) - 0.5, n_twins, f" all {n_twins} pairs", va="bottom", ha="right", fontsize=9,
                 color=MUTED)
    ax_twin.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_twin.set(ylim=(0, n_twins + 2), xlabel="", ylabel="twins told apart",
                title="C · Harmful vs benign twins told apart")
    ax_twin.tick_params(axis="x", rotation=25)

    # D. AUC vs latency
    sns.scatterplot(data=df, x="latency", y="auc", hue="model", style="config", palette=PALETTE, markers=MARKERS,
                    hue_order=models, style_order=configs, s=150, edgecolor="white", linewidth=0.8, ax=ax_lat,
                    legend=False, zorder=3)
    ax_lat.set_xscale("log")
    ax_lat.set(xlabel="median latency per prompt, ms (log; Laya on MPS, GLiNER on CPU)", ylabel="ROC-AUC",
               title="D · Accuracy vs cost")
    ax_lat.set_xticks([100, 200, 500, 1000])
    ax_lat.set_xticklabels(["100", "200", "500", "1000"])

    handles = [Line2D([], [], marker="o", linestyle="", markersize=11, color=PALETTE[m], label=m) for m in models]
    handles += [Line2D([], [], linestyle="", label=" ")]
    handles += [Line2D([], [], marker=MARKERS[c], linestyle="", markersize=10, color=INK, label=c) for c in configs]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.005))
    n = {k: sum(r["kind"] == k for r in test) for k in ("harmful", "benign", "hard_negative")}
    title = "every mode and combination" if set_name == "core" else "out-of-distribution public set"
    how = "thresholds tuned on dev" if set_name == "core" else "thresholds carried over from the core set's dev split"
    fig.suptitle(f"reflexguard eval · GLiNER2.5-Decide vs Laya · {title}", fontsize=16, fontweight="bold", color=INK,
                 y=0.995)
    fig.text(0.5, 0.962, f"{set_name} set: {len(test)} prompts ({n['harmful']} harmful, {n['benign']} benign, "
             f"{n['hard_negative']} hard negatives) · {how}", ha="center", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    path = out / "overview.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def per_category(results: list[dict], test: list[dict], categories: dict[str, str], out: Path) -> Path:
    counts = {c: sum(c in r["labels"] for r in test) for c in categories}
    cols = [c for c in categories if counts[c]]
    data = pd.DataFrame(
        [[r["tuned"]["per_category"][c] for c in cols] for r in results],
        index=[f"{MODEL_NAMES.get(r['model'], r['model'])} · {r['config']}" for r in results],
        columns=[f"{c.replace('_', ' ')}\nn={counts[c]}" for c in cols],
    )
    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(data) + 2))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.5, vmax=1.0, linewidths=1.5, linecolor="white",
                cbar_kws={"label": "ROC-AUC", "shrink": 0.6}, annot_kws={"fontsize": 9}, ax=ax)
    for i, label in enumerate(data.index):  # a thin rule between models
        if i and label.split(" · ")[0] != data.index[i - 1].split(" · ")[0]:
            ax.axhline(i, color=INK, linewidth=2)
    ax.set(xlabel="", ylabel="")
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    thin = [c.replace("_", " ") for c in cols if counts[c] < 10]
    note = f"fewer than 10 labeled rows (indicative only): {', '.join(thin)}" if thin else "every category has 10+ rows"
    ax.set_title("Per-category ROC-AUC · each category's own score, labeled rows vs all non-harmful rows\n" + note,
                 fontsize=12, color=INK, loc="left")
    fig.tight_layout()
    path = out / "per_category.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot(results: list[dict], rows: list[dict], test: list[dict], categories: dict[str, str], out: Path,
         name: str = "core") -> None:
    out.mkdir(parents=True, exist_ok=True)
    for path in (overview(results, test, out, set_name=name), per_category(results, test, categories, out)):
        print(f"wrote {path}")
