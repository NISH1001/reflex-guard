"""Figures for scripts/eval.py: `eval.py plot` (per set) and `eval.py paper` (publication figures).

One style table, STYLE, gives every model the same colour, hatch and marker in every figure.
Colours are Okabe–Ito hues in an order validated for colour-vision deficiency (worst adjacent
CVD ΔE 9.6); hatches and markers repeat the identity for greyscale print, and every bar gets a
dark edge so the lighter hues keep an outline. Every figure is written as PDF (fonts embedded)
and PNG.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# model -> (label, colour, hatch, marker, where it ran)
STYLE = {
    "laya": ("Laya", "#009E73", "///", "o", "laptop GPU"),
    "gliner:fp32:orders=3": ("GLiNER fp32", "#D55E00", "\\\\\\", "s", "CPU"),
    "gliner:int8:orders=3": ("GLiNER int8", "#0072B2", "xxx", "D", "CPU"),
    "jev": ("Jev 1.14", "#CC79A7", "...", "^", "API"),
}
# The NOUL label-order ablation reuses the precision's colour: fp32 vermillion, int8 blue.
ABLATION = {
    "fp32": ("GLiNER fp32", "#D55E00", "\\\\\\", ["gliner:fp32", "gliner:fp32:orders=2", "gliner:fp32:orders=3"]),
    "int8": ("GLiNER int8", "#0072B2", "xxx", ["gliner:int8", None, "gliner:int8:orders=3"]),
}
CONFIGS = ["noul", "choice", "score", "any", "all", "votes=2"]
INK, MUTED, GRID = "#1f1f1d", "#5f5f5b", "#dcdcd8"
SEQUENTIAL = "Blues"  # one hue, light -> dark, for heatmaps


def _style() -> None:
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "hatch.linewidth": 0.6, "hatch.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType, as journals ask
    })


def _save(fig, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _models(*result_sets: list[dict]) -> list[str]:
    present = {r["model"] for results in result_sets for r in results}
    return [m for m in STYLE if m in present]


def _legend(fig, models: list[str], y: float, extra: list | None = None, where: bool = False) -> None:
    handles = [Patch(facecolor=STYLE[m][1], hatch=STYLE[m][2], edgecolor=INK, linewidth=0.5,
                     label=STYLE[m][0] + (f" ({STYLE[m][4]})" if where else "")) for m in models]
    handles += extra or []
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, y),
               handlelength=2.2, handleheight=1.1, columnspacing=1.4)


def _bars(ax, results: list[dict], metric, models: list[str], configs: list[str] = CONFIGS, err=None) -> None:
    """Grouped bars: one group per config, one hatched bar per model, in the same slot on every panel."""
    width = 0.8 / len(models)
    for j, model in enumerate(models):
        _, color, hatch, _, _ = STYLE[model]
        rows = [next((r for r in results if r["model"] == model and r["config"] == c), None) for c in configs]
        values = [metric(r) if r else np.nan for r in rows]
        xs = [i - 0.4 + width * (j + 0.5) for i in range(len(configs))]
        ax.bar(xs, values, width * 0.9, color=color, hatch=hatch, edgecolor=INK, linewidth=0.5, zorder=3)
        if err is not None:
            lo = [v - err(r)[0] if r else 0 for v, r in zip(values, rows)]
            hi = [err(r)[1] - v if r else 0 for v, r in zip(values, rows)]
            ax.errorbar(xs, values, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="none", ecolor=INK,
                        elinewidth=0.6, capsize=1.5, zorder=4)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs)
    _grid(ax)


def _fp_rate(r: dict) -> float:
    m = r["tuned"]
    nb, nh = r.get("n_benign", 1), r.get("n_hard", 1)
    return (m["benign_fp"] * nb + m["hard_fp"] * nh) / (nb + nh)


def _roc(scored: list[tuple[float, bool]]) -> tuple[list[float], list[float]]:
    pos = sum(y for _, y in scored)
    neg = len(scored) - pos
    fpr, tpr, tp, fp = [0.0], [0.0], 0, 0
    for _, y in sorted(scored, key=lambda x: -x[0]):
        tp, fp = tp + y, fp + (not y)
        fpr.append(fp / neg)
        tpr.append(tp / pos)
    return fpr, tpr


# ── per-set figures (eval.py plot) ─────────────────────────────────────────
def overview(results: list[dict], test: list[dict], out: Path, set_name: str) -> list[Path]:
    """Four panels: AUC by config, ROC of each model's best config, recall vs FP, and twins / hard negatives."""
    _style()
    models = _models(results)
    nb, nh = sum(r["kind"] == "benign" for r in test), sum(r["kind"] == "hard_negative" for r in test)
    results = [r | {"n_benign": nb, "n_hard": nh} for r in results]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    (ax_auc, ax_roc), (ax_op, ax_c) = axes

    _bars(ax_auc, results, lambda r: r["tuned"]["auc"], models)
    ax_auc.axhline(0.5, color=INK, linestyle=(0, (3, 2)), linewidth=0.7, zorder=4)
    ax_auc.set(ylim=(0, 1), ylabel="ROC-AUC", title="A · ROC-AUC by config (dashed: chance)")

    for model in models:
        best = max((r for r in results if r["model"] == model), key=lambda r: r["tuned"]["auc"])
        fpr, tpr = _roc(best["test_scores"])
        ax_roc.plot(fpr, tpr, color=STYLE[model][1], linewidth=1.6,
                    label=f"{STYLE[model][0]} · {best['config']} ({best['tuned']['auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle=(0, (3, 2)), color=INK, linewidth=0.7)
    ax_roc.set(xlim=(0, 1), ylim=(0, 1), xlabel="false-positive rate", ylabel="true-positive rate",
               title="B · ROC, best config per model")
    ax_roc.legend(loc="lower right", frameon=False, fontsize=6.5)
    _grid(ax_roc, "both")

    for model in models:
        rs = [r for r in results if r["model"] == model and r["config"] in CONFIGS]
        ax_op.scatter([_fp_rate(r) for r in rs], [r["tuned"]["recall"] for r in rs], s=26, color=STYLE[model][1],
                      marker=STYLE[model][3], edgecolor=INK, linewidth=0.5, zorder=3)
    how = "dev-tuned threshold" if set_name == "core" else "core-dev threshold"
    ax_op.set(xlim=(0, 1), ylim=(0, 1), xlabel="false-positive rate (benign + hard negatives)",
              ylabel="harmful recall", title=f"C · Recall vs false positives ({how})")
    ax_op.text(0.02, 0.97, "better: up and left", fontsize=6.5, color=MUTED, va="top", transform=ax_op.transAxes)
    _grid(ax_op, "both")

    n_twins = max(r["tuned"]["pairs"][1] for r in results)
    if n_twins:
        _bars(ax_c, results, lambda r: r["tuned"]["pairs"][0], models)
        ax_c.axhline(n_twins, color=INK, linestyle=":", linewidth=0.7)
        ax_c.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_c.set(ylim=(0, n_twins + 1), ylabel=f"twins told apart (of {n_twins})",
                 title="D · Harmful vs benign twins told apart")
    else:
        _bars(ax_c, results, lambda r: r["tuned"]["hard_fp"], models)
        ax_c.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax_c.set(ylim=(0, 1), ylabel="hard negatives flagged", title="D · Benign look-alikes flagged (lower is better)")
    for ax in (ax_auc, ax_c):
        ax.tick_params(axis="x", labelsize=6.5)

    markers = [Line2D([], [], marker=STYLE[m][3], linestyle="", color=STYLE[m][1], markeredgecolor=INK,
                      markersize=5, label=f"{STYLE[m][0]} (C)") for m in models]
    _legend(fig, models, 1.05, extra=markers)
    title = "core set, test split" if set_name == "core" else "out-of-distribution set"
    fig.suptitle(f"reflexguard eval · {title} · {len(test)} prompts", y=1.09, fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out / "overview")


def per_category(results: list[dict], test: list[dict], categories: dict[str, str], out: Path) -> list[Path]:
    _style()
    counts = {c: sum(c in r["labels"] for r in test) for c in categories}
    cols = [c for c in categories if counts[c]]
    rows = [r for r in results if r["config"] in CONFIGS]
    data = np.array([[r["tuned"]["per_category"][c] or np.nan for c in cols] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 0.17 * len(rows) + 1.2))
    im = ax.imshow(data, cmap=SEQUENTIAL, vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if data[i, j] > 0.8 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c.replace('_', ' ')}\nn={counts[c]}" for c in cols], fontsize=6.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{STYLE[r['model']][0]} · {r['config']}" for r in rows], fontsize=6.5)
    for i, r in enumerate(rows):  # a white rule between models
        if i and r["model"] != rows[i - 1]["model"]:
            ax.axhline(i - 0.5, color="white", linewidth=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    bar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
    bar.set_label("ROC-AUC (each category's own score)", fontsize=7)
    thin = [c.replace("_", " ") for c in cols if counts[c] < 10]
    ax.set_title("Per-category ROC-AUC" + (f" · fewer than 10 rows (indicative): {', '.join(thin)}" if thin else ""),
                 fontsize=8, loc="left")
    fig.tight_layout()
    return _save(fig, out / "per_category")


def plot(results: list[dict], rows: list[dict], test: list[dict], categories: dict[str, str], out: Path,
         name: str = "core") -> None:
    out.mkdir(parents=True, exist_ok=True)
    keep = [r for r in results if r["model"] in STYLE]
    for path in overview(keep, test, out, set_name=name) + per_category(keep, test, categories, out):
        print(f"wrote {path}")


# ── paper figures (eval.py paper) ──────────────────────────────────────────
def paper(core: list[dict], ood: list[dict], out: Path) -> list[Path]:
    _style()
    out.mkdir(parents=True, exist_ok=True)
    models = _models(core, ood)
    chance = Line2D([], [], color=INK, linestyle=(0, (3, 2)), linewidth=0.7, label="chance")
    paths = []

    # 1. ROC-AUC by config, core vs ood
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, results, title in [(axes[0], core, "Core set (test split)"), (axes[1], ood, "Out-of-distribution set")]:
        _bars(ax, results, lambda r: r["tuned"]["auc"], models)
        ax.axhline(0.5, color=INK, linestyle=(0, (3, 2)), linewidth=0.7, zorder=4)
        ax.set(title=title, ylim=(0, 1))
    axes[0].set_ylabel("ROC-AUC")
    _legend(fig, models, 1.07, extra=[chance])
    fig.tight_layout()
    paths += _save(fig, out / "paper_auc")

    # 2. recall and false-positive rate at the core-dev threshold, both sets
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True, sharey=True)
    for row, (results, set_name) in enumerate([(core, "core"), (ood, "out-of-distribution")]):
        for col, (metric, label) in enumerate([(lambda r: r["tuned"]["recall"], "Harmful recall"),
                                                (_fp_rate, "False-positive rate (lower is better)")]):
            ax = axes[row][col]
            _bars(ax, results, metric, models)
            ax.set(ylim=(0, 1), title=f"{label} · {set_name}")
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _legend(fig, models, 1.04)
    fig.text(0.5, -0.015, "Each config's threshold is tuned on the core dev split and applied unchanged to both sets; "
             "false positives count benign and hard-negative prompts.", ha="center", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    paths += _save(fig, out / "paper_rates")

    # 3. latency per prompt: median bar, p10–p90 whisker
    groups = {"noul": "noul", "choice": "choice", "score": "score", "all three modes": "any"}
    lat = [r | {"config": label} for r in core for label, cfg in groups.items() if r["config"] == cfg]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    _bars(ax, lat, lambda r: statistics.median(r["latencies"]), models, list(groups),
          err=lambda r: (float(np.percentile(r["latencies"], 10)), float(np.percentile(r["latencies"], 90))))
    ax.set(ylabel="latency per prompt (ms)", title="Latency per prompt, core test split (median, p10–p90)")
    ax.set_ylim(bottom=0)
    _legend(fig, models, 1.1, where=True)
    fig.text(0.5, -0.04, "Not like-for-like hardware: Laya ran on the laptop GPU, GLiNER on its CPU, Jev over the "
             "network. Modes are asked in separate calls; 'all three' sums them. Local timings were taken with other "
             "jobs loading the machine.", ha="center", fontsize=6.5, color=MUTED, wrap=True)
    fig.tight_layout()
    paths += _save(fig, out / "paper_latency")

    # 4. accuracy vs cost
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, results, title in [(axes[0], core, "Core set"), (axes[1], ood, "Out-of-distribution set")]:
        for model in models:
            rs = [r for r in results if r["model"] == model and r["config"] in CONFIGS]
            ax.scatter([statistics.median(r["latencies"]) for r in rs], [r["tuned"]["auc"] for r in rs], s=30,
                       color=STYLE[model][1], marker=STYLE[model][3], edgecolor=INK, linewidth=0.5, zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set(title=title, xlabel="median latency per prompt (ms, log)", ylim=(0.5, 1))
        _grid(ax, "both")
    axes[0].set_ylabel("ROC-AUC (one point per config)")
    markers = [Line2D([], [], marker=STYLE[m][3], linestyle="", color=STYLE[m][1], markeredgecolor=INK,
                      markersize=5.5, label=STYLE[m][0]) for m in models]
    fig.legend(handles=markers, loc="upper center", ncol=len(markers), frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    paths += _save(fig, out / "paper_tradeoff")

    # 5. ablation: GLiNER NOUL with 1, 2, 3 label orders
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    panels = [("NOUL AUC · core", core, lambda r: r["tuned"]["auc"], "noul"),
              ("NOUL AUC · out-of-distribution", ood, lambda r: r["tuned"]["auc"], "noul"),
              ("'any' benign FP · core (lower = better)", core, lambda r: r["tuned"]["benign_fp"], "any")]
    for ax, (title, results, metric, config) in zip(axes, panels):
        width = 0.38
        for j, (key, (label, color, hatch, variants)) in enumerate(ABLATION.items()):
            values = [next((metric(r) for r in results if r["model"] == v and r["config"] == config), np.nan)
                      if v else np.nan for v in variants]
            ax.bar([i - 0.2 + width * j for i in range(3)], values, width * 0.9, color=color, hatch=hatch,
                   edgecolor=INK, linewidth=0.5, zorder=3, label=label)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["1 order", "2 orders", "3 orders"])
        ax.set(title=title, ylim=(0, 1 if "AUC" in title else 0.5))
        if "FP" in title:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _grid(ax)
    handles = [Patch(facecolor=c, hatch=h, edgecolor=INK, linewidth=0.5, label=l) for l, c, h, _ in ABLATION.values()]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08),
               handlelength=2.2, handleheight=1.1)
    fig.text(0.5, -0.04, "noul_orders=k asks NOUL's multi-label question k times with the categories rotated and "
             "averages; gaps are runs not made.", ha="center", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    paths += _save(fig, out / "paper_noul_orders")
    return paths
