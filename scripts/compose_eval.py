"""Offline evaluation of EnsembleGuard and CascadeGuard from the cached per-model scores.

    uv run python scripts/compose_eval.py        # -> docs/experiments/11-tables.md + paper_cascades figure

No model is called. Each model uses one config, the one with the best balanced accuracy on the
core dev split, at its dev-tuned threshold. Ensembles and cascades then combine those members'
per-category decisions exactly as reflexguard.compose does, and are scored on the core test
split and on the OOD set (thresholds carried over from core dev).
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import eval as ev  # scripts/eval.py

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "experiments"
FIGURES = ROOT / "data" / "eval" / "figures"
MODELS = {"laya": "Laya", "gliner:fp32:orders=3": "GLiNER", "jev": "Jev", "openai": "GPT-6 Luna"}


def member(model: str) -> dict:
    """One model's per-prompt category scores, threshold and latency, for both sets."""
    out = {}
    best = None
    for name in ("core", "ood"):
        ev.use_set(name)
        rows, categories = ev.load()
        cache = ev.read_cache(model)
        per_mode = {m: {} for m in ev.MODES}
        lat = {m: {} for m in ev.MODES}
        chunks = {}
        for r in rows:
            for m in ev.MODES:
                e = cache[ev.key(model, m, categories, r["prompt"])]
                per_mode[m][r["id"]], lat[m][r["id"]] = e["scores"], e["latency_ms"]
            e = cache.get(ev.key(model, ev.PER_CHUNK, categories, r["prompt"]))
            if e:
                chunks[r["id"]] = e["chunks"]

        def scores_for(config):
            expr, needs = ev.CONFIGS[config]

            def combined(r, c):
                if len(needs) > 1 and r["id"] in chunks:
                    return max(expr.combine({m: ch[m][c] for m in needs}) for ch in chunks[r["id"]])
                return expr.combine({m: per_mode[m][r["id"]][c] for m in needs})

            return {r["id"]: {c: combined(r, c) for c in categories} for r in rows}

        if best is None:  # pick the config on core dev only
            dev = [r for r in rows if r["split"] == "dev"]
            candidates = []
            for config in ev.CONFIGS:
                s = scores_for(config)
                scored = [(max(s[r["id"]].values()), r["kind"] == "harmful") for r in dev]
                t = ev.best_threshold(scored)
                pos = sum(y for _, y in scored)
                tpr = sum(v >= t and y for v, y in scored) / pos
                tnr = sum(v < t and not y for v, y in scored) / (len(scored) - pos)
                candidates.append(((tpr + tnr) / 2, config, t))
            _, config, threshold = max(candidates)
            best = (config, threshold)
        config, threshold = best
        needs = ev.CONFIGS[config][1]
        out[name] = {"rows": rows, "scores": scores_for(config),
                     "latency": {r["id"]: sum(lat[m][r["id"]] for m in needs) for r in rows}}
    return {"config": config, "threshold": threshold, **out}


def metrics(rows: list[dict], flagged: dict[str, bool], latency: dict[str, float], escalated=None) -> dict:
    test = [r for r in rows if r["split"] == "test"]
    harm = [r for r in test if r["kind"] == "harmful"]
    ben = [r for r in test if r["kind"] == "benign"]
    hard = [r for r in test if r["kind"] == "hard_negative"]
    pairs = {}
    for r in test:
        if r["pair_id"]:
            pairs.setdefault(r["pair_id"], {})[r["kind"]] = r["id"]
    twins = [p for p in pairs.values() if "harmful" in p and "hard_negative" in p]
    out = {
        "recall": sum(flagged[r["id"]] for r in harm) / len(harm),
        "benign_fp": sum(flagged[r["id"]] for r in ben) / len(ben),
        "hard_fp": sum(flagged[r["id"]] for r in hard) / len(hard),
        "twins": f"{sum(flagged[p['harmful']] and not flagged[p['hard_negative']] for p in twins)}/{len(twins)}"
                 if twins else "—",
        "latency": statistics.median(latency[r["id"]] for r in test),
    }
    if escalated is not None:
        out["escalated"] = sum(escalated[r["id"]] for r in test) / len(test)
    return out


def ensemble(members: list[dict], k: int, set_name: str) -> dict:
    """Votes(k) per category over the members' decisions (k=1: AnyOf, k=n: AllOf)."""
    rows = members[0][set_name]["rows"]
    flagged, latency = {}, {}
    for r in rows:
        i = r["id"]
        cats = members[0][set_name]["scores"][i]
        flagged[i] = any(sum(m[set_name]["scores"][i][c] >= m["threshold"] for m in members) >= k for c in cats)
        latency[i] = max(m[set_name]["latency"][i] for m in members)  # members run concurrently
    return metrics(rows, flagged, latency)


def escalates(m: dict, top: float, margin: tuple[float, float]) -> bool:
    """Margin(below, above) on one model's verdict: its top score near its own threshold."""
    below, above = margin
    return m["threshold"] - below <= top < m["threshold"] + above


def cascade(stages: list[dict], margins: list[tuple[float, float]], set_name: str) -> dict:
    """CascadeGuard(stages, rule=[Margin(*m) for m in margins]): each stage answers unless its rule escalates."""
    rows = stages[0][set_name]["rows"]
    flagged, latency, escalated = {}, {}, {}
    for r in rows:
        i = r["id"]
        latency[i] = 0.0
        for n, stage in enumerate(stages):
            top = max(stage[set_name]["scores"][i].values())
            latency[i] += stage[set_name]["latency"][i]
            if n == len(stages) - 1 or not escalates(stage, top, margins[n]):
                break
        flagged[i] = top >= stage["threshold"]
        escalated[i] = n > 0
    return metrics(rows, flagged, latency, escalated)


def single(m: dict, set_name: str) -> dict:
    rows = m[set_name]["rows"]
    flagged = {r["id"]: max(m[set_name]["scores"][r["id"]].values()) >= m["threshold"] for r in rows}
    return metrics(rows, flagged, m[set_name]["latency"])


def row(label: str, core: dict, ood: dict) -> str:
    esc = lambda d: f" | {d['escalated']:.0%}" if "escalated" in d else ""
    return (f"| {label} | {core['recall']:.1%} | {core['benign_fp']:.1%} | {core['hard_fp']:.1%} | {core['twins']} "
            f"| {ood['recall']:.1%} | {ood['benign_fp']:.1%} | {ood['hard_fp']:.1%} | {core['latency']:.0f} ms"
            + (f" | {core['escalated']:.0%} / {ood['escalated']:.0%}" if "escalated" in core else "") + " |")


def main() -> None:
    m = {key: member(key) for key in MODELS}
    for key, v in m.items():
        print(f"{MODELS[key]}: config {v['config']} at {v['threshold']:.3f} (best dev balanced accuracy)", file=sys.stderr)
    header = ("| guard | core recall | core benign FP | core hard-neg FP | twins | OOD recall | OOD benign FP "
              "| OOD hard-neg FP | median latency |")
    lines = ["## Single models (each at its dev-picked config)", "", header, "|" + "---|" * 9]
    for key in MODELS:
        lines.append(row(MODELS[key], single(m[key], "core"), single(m[key], "ood")))

    ens = [
        ("Laya \\| Jev (AnyOf)", ["laya", "jev"], 1), ("Laya & Jev (AllOf)", ["laya", "jev"], 2),
        ("Jev \\| GPT-6 Luna (AnyOf)", ["jev", "openai"], 1), ("Jev & GPT-6 Luna (AllOf)", ["jev", "openai"], 2),
        ("Votes(2): Laya, GLiNER, Jev", ["laya", "gliner:fp32:orders=3", "jev"], 2),
        ("Votes(2): Laya, Jev, GPT-6 Luna", ["laya", "jev", "openai"], 2),
        ("Votes(3): all four", list(MODELS), 3),
        ("Laya \\| GLiNER (local only)", ["laya", "gliner:fp32:orders=3"], 1),
    ]
    lines += ["", "## Ensembles", "", header, "|" + "---|" * 9]
    for label, keys, k in ens:
        members = [m[x] for x in keys]
        lines.append(row(label, ensemble(members, k, "core"), ensemble(members, k, "ood")))

    sym = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    asym = [(0.1, 0.02), (0.2, 0.05), (0.3, 0.1)]
    lines += ["", "## Cascades (`Margin(below, above)` around the first guard's own threshold)", "",
              header[:-1] + " | escalated (core / OOD) |", "|" + "---|" * 10]
    curves = {}
    for first, second in [("laya", "jev"), ("laya", "openai"), ("gliner:fp32:orders=3", "jev")]:
        for margin in [(w, w) for w in sym] + asym:
            core = cascade([m[first], m[second]], [margin], "core")
            ood = cascade([m[first], m[second]], [margin], "ood")
            name = f"Margin({margin[0]})" if margin[0] == margin[1] else f"Margin({margin[0]}, {margin[1]})"
            lines.append(row(f"{MODELS[first]} → {MODELS[second]}, {name}", core, ood))
            if margin[0] == margin[1]:
                curves.setdefault((first, second), []).append((margin, core, ood))
    three = [m["gliner:fp32:orders=3"], m["laya"], m["jev"]]
    for margins in [[(0.05, 0.05), (0.2, 0.2)], [(0.05, 0.02), (0.3, 0.1)]]:
        core, ood = cascade(three, margins, "core"), cascade(three, margins, "ood")
        name = ", ".join(f"Margin{mg}" for mg in margins)
        lines.append(row(f"GLiNER → Laya → Jev, [{name}]", core, ood))

    (OUT / "11-tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(m, curves)


def plot(m: dict, curves: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_eval import INK, STYLE, _save, _style

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, set_name, title in [(axes[0], "core", "Core set (test split)"), (axes[1], "ood", "Out-of-distribution set")]:
        for (first, second), pts in curves.items():
            color, marker = STYLE[second][1], STYLE[first][3]
            data = [(p[1 if set_name == "core" else 2]) for p in pts]
            ax.plot([d["escalated"] for d in data], [d["recall"] for d in data], color=color, marker=marker,
                    markeredgecolor=INK, linewidth=1.2, markersize=5,
                    label=f"{STYLE[first][0]} → {STYLE[second][0]}")
            ref = single(m[second], set_name)
            ax.axhline(ref["recall"], color=color, linestyle=(0, (3, 2)), linewidth=0.8)
        ax.set(title=title, xlabel="share of prompts escalated to the second guard", xlim=(0, 1), ylim=(0.5, 1))
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(True, color="#dcdcd8", linewidth=0.5)
    axes[0].set_ylabel("harmful recall")
    axes[0].legend(loc="lower right", frameon=False, fontsize=6.5)
    fig.text(0.5, -0.04, "Each line sweeps a symmetric Margin(w), w = 0.02 → 0.3, around the first guard's threshold; "
             "dashed: the second guard alone on every prompt.", ha="center", fontsize=6.5, color="#5f5f5b")
    fig.tight_layout()
    for p in _save(fig, FIGURES / "paper_cascades"):
        print(f"wrote {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
