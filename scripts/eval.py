"""Benchmark reflexguard models × modes × combinations on data/eval/prompts.csv.

    uv run python scripts/eval.py run --models laya,gliner:fp32,gliner:int8
    uv run python scripts/eval.py report          # no model calls: reads the cache

`run` asks each model one mode at a time and caches every prompt's per-category scores
in data/eval/cache/<model>.jsonl, keyed by model, mode, category set and prompt, so a
re-run only scores what is new. `report` combines the cached per-mode scores with
reflexguard's own mode expressions (any / all / votes), picks each config's threshold on
the dev split, and reports on the test split -> data/eval/results.md and results.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from reflexguard import GlinerGuard, Guard, LayaGuard
from reflexguard.modes import with_votes

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "eval"
CACHE = DATA / "cache"
MODES = {"noul": Guard.NOUL, "choice": Guard.CHOICE, "score": Guard.SCORE}
CONFIGS = {  # name -> (expression over the three modes, modes it needs)
    "noul": (Guard.NOUL, ["noul"]),
    "choice": (Guard.CHOICE, ["choice"]),
    "score": (Guard.SCORE, ["score"]),
    "any": (Guard.NOUL | Guard.CHOICE | Guard.SCORE, ["noul", "choice", "score"]),
    "all": (Guard.NOUL & Guard.CHOICE & Guard.SCORE, ["noul", "choice", "score"]),
    "votes=2": (with_votes(Guard.NOUL | Guard.CHOICE | Guard.SCORE, 2), ["noul", "choice", "score"]),
    "noul|choice": (Guard.NOUL | Guard.CHOICE, ["noul", "choice"]),
    "noul&choice": (Guard.NOUL & Guard.CHOICE, ["noul", "choice"]),
}


def load() -> tuple[list[dict], dict[str, str]]:
    rows = list(csv.DictReader(open(DATA / "prompts.csv")))
    for r in rows:
        r["labels"] = set(filter(None, r["categories"].split(";")))
    return rows, json.loads((DATA / "categories.json").read_text())


def key(model: str, mode: str, categories: dict[str, str], prompt: str) -> str:
    payload = json.dumps([model, mode, categories, prompt], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def read_cache(model: str) -> dict[str, dict]:
    path = CACHE / f"{model.replace(':', '_')}.jsonl"
    if not path.exists():
        return {}
    return {e["key"]: e for e in map(json.loads, path.read_text().splitlines())}


def make_guard(model: str, mode: str, categories: dict[str, str], shared: dict):
    if model == "laya":
        if "router" not in shared:
            from laya import Router

            shared["router"] = Router()
            shared["router"].preload(["english"])
        return LayaGuard(categories=categories, mode=MODES[mode], router=shared["router"])
    name, _, precision = model.partition(":")
    if name != "gliner":
        raise SystemExit(f"unknown model {model!r}; use laya or gliner:<fp32|int8|fp16>")
    if "runtime" not in shared:
        from reflexguard.impls.gliner import load_runtime

        shared["runtime"] = load_runtime(precision=precision or "fp32")
    return GlinerGuard(categories=categories, mode=MODES[mode], precision=precision or "fp32",
                       runtime=shared["runtime"])


def run(models: list[str], limit: int | None) -> None:
    rows, categories = load()
    rows = rows[:limit] if limit else rows
    CACHE.mkdir(parents=True, exist_ok=True)
    for model in models:
        cache, shared = read_cache(model), {}
        path = CACHE / f"{model.replace(':', '_')}.jsonl"
        with open(path, "a") as out:
            for mode in MODES:
                todo = [r for r in rows if key(model, mode, categories, r["prompt"]) not in cache]
                print(f"{model} {mode}: {len(todo)} to score, {len(rows) - len(todo)} cached", file=sys.stderr)
                if not todo:
                    continue
                guard = make_guard(model, mode, categories, shared)
                guard.guard("warm up")
                for i, r in enumerate(todo, 1):
                    t0 = time.perf_counter()
                    res = guard.guard(r["prompt"])
                    ms = (time.perf_counter() - t0) * 1000
                    entry = {"key": key(model, mode, categories, r["prompt"]), "model": model, "mode": mode,
                             "id": r["id"], "scores": {c.name: round(c.by_mode[mode], 5) for c in res.ranked},
                             "latency_ms": round(ms, 1)}
                    out.write(json.dumps(entry) + "\n")
                    out.flush()
                    if i % 100 == 0:
                        print(f"  {i}/{len(todo)}", file=sys.stderr)


# ── metrics ─────────────────────────────────────────────────────────────────
def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):  # average ranks over ties
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += sum(1 for k in range(i, j) if ranked[k][1]) * (i + j + 1) / 2
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def best_threshold(scored: list[tuple[float, bool]]) -> float:
    """Threshold maximising balanced accuracy on dev (ties -> the higher threshold)."""
    pos = sum(y for _, y in scored)
    neg = len(scored) - pos
    best, best_t = -1.0, 0.5
    for t in sorted({s for s, _ in scored}, reverse=True):
        tpr = sum(s >= t and y for s, y in scored) / pos
        tnr = sum(s < t and not y for s, y in scored) / neg
        if (tpr + tnr) / 2 > best:
            best, best_t = (tpr + tnr) / 2, t
    return best_t


def evaluate(rows, categories, scores, lat, config, threshold):
    """scores: id -> {category: combined score}. Metrics on the given rows at `threshold`."""
    top = {r["id"]: max(scores[r["id"]].values()) for r in rows}
    flagged = {i: s >= threshold for i, s in top.items()}
    harm = [r for r in rows if r["kind"] == "harmful"]
    ben = [r for r in rows if r["kind"] == "benign"]
    hard = [r for r in rows if r["kind"] == "hard_negative"]
    pairs: dict[str, dict[str, str]] = {}
    for r in rows:
        if r["pair_id"]:
            pairs.setdefault(r["pair_id"], {})[r["kind"]] = r["id"]
    pair_ok = [p for p in pairs.values() if "harmful" in p and "hard_negative" in p]
    out = {
        "recall": sum(flagged[r["id"]] for r in harm) / len(harm),
        "benign_fp": sum(flagged[r["id"]] for r in ben) / len(ben),
        "hard_fp": sum(flagged[r["id"]] for r in hard) / len(hard),
        "auc": auc([top[r["id"]] for r in harm], [top[r["id"]] for r in ben + hard]),
        "pairs": (sum(flagged[p["harmful"]] and not flagged[p["hard_negative"]] for p in pair_ok), len(pair_ok)),
        "latency": statistics.median(sum(lat[m][r["id"]] for m in CONFIGS[config][1]) for r in rows),
    }
    out["per_category"] = {
        c: auc([scores[r["id"]][c] for r in rows if c in r["labels"]],
               [scores[r["id"]][c] for r in rows if r["kind"] != "harmful"])
        for c in categories
    }
    out["by_domain"] = {}
    for d in sorted({r["domain"] for r in rows}):
        sub = [r for r in rows if r["domain"] == d]
        h = [r for r in sub if r["kind"] == "harmful"]
        b = [r for r in sub if r["kind"] != "harmful"]
        out["by_domain"][d] = (sum(flagged[r["id"]] for r in h), len(h), sum(flagged[r["id"]] for r in b), len(b))
    return out


def report() -> None:
    rows, categories = load()
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    models = sorted(p.stem.replace("_", ":", 1) for p in CACHE.glob("*.jsonl"))
    results = []
    for model in models:
        cache = read_cache(model)
        per_mode: dict[str, dict[str, dict[str, float]]] = {m: {} for m in MODES}
        lat: dict[str, dict[str, float]] = {m: {} for m in MODES}
        for r in rows:
            for m in MODES:
                e = cache.get(key(model, m, categories, r["prompt"]))
                if e:
                    per_mode[m][r["id"]], lat[m][r["id"]] = e["scores"], e["latency_ms"]
        for config, (expr, needs) in CONFIGS.items():
            if any(len(per_mode[m]) < len(rows) for m in needs):
                print(f"skip {model} {config}: not every row is cached", file=sys.stderr)
                continue
            scores = {r["id"]: {c: expr.combine({m: per_mode[m][r["id"]][c] for m in needs}) for c in categories}
                      for r in rows}
            t = best_threshold([(max(scores[r["id"]].values()), r["kind"] == "harmful") for r in dev])
            results.append({"model": model, "config": config, "threshold": t,
                            "at_05": evaluate(test, categories, scores, lat, config, 0.5),
                            "tuned": evaluate(test, categories, scores, lat, config, t)})
    write_report(results, rows, test, categories)


def pct(x: float) -> str:
    return f"{x:.1%}"


def write_report(results, rows, test, categories) -> None:
    kinds = {k: sum(r["kind"] == k for r in test) for k in ("harmful", "benign", "hard_negative")}
    lines = [
        "# reflexguard eval",
        "",
        f"Test split: {len(test)} of {len(rows)} prompts ({kinds['harmful']} harmful, {kinds['benign']} benign, "
        f"{kinds['hard_negative']} hard negatives). Thresholds are picked on the dev split (best balanced accuracy) "
        "and applied to test. A prompt is flagged when any category's combined score reaches the threshold. "
        "Latency is the median per prompt, summed over the modes a config needs (each mode is its own call).",
        "",
        "## Dev-tuned threshold",
        "",
        "| model | config | threshold | recall | benign FP | hard-negative FP | AUC | pairs told apart | latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r["tuned"]
        lines.append(f"| {r['model']} | {r['config']} | {r['threshold']:.3f} | {pct(m['recall'])} | {pct(m['benign_fp'])} "
                     f"| {pct(m['hard_fp'])} | {m['auc']:.3f} | {m['pairs'][0]}/{m['pairs'][1]} | {m['latency']:.0f} ms |")
    lines += ["", "## Fixed threshold 0.5", "",
              "| model | config | recall | benign FP | hard-negative FP | pairs told apart |", "|---|---|---|---|---|---|"]
    for r in results:
        m = r["at_05"]
        lines.append(f"| {r['model']} | {r['config']} | {pct(m['recall'])} | {pct(m['benign_fp'])} | {pct(m['hard_fp'])} "
                     f"| {m['pairs'][0]}/{m['pairs'][1]} |")
    counts = {c: sum(c in r["labels"] for r in test) for c in categories}
    lines += ["", "## Per-category AUC (test)", "",
              "Each category's own score: rows labeled with it vs every non-harmful row. "
              "`n` is the number of test rows labeled with the category.", "",
              "| model | config | " + " | ".join(f"{c} (n={counts[c]})" for c in categories) + " |",
              "|---|---|" + "---|" * len(categories)]
    for r in results:
        pc = r["tuned"]["per_category"]
        lines.append(f"| {r['model']} | {r['config']} | " + " | ".join("—" if pc[c] is None else f"{pc[c]:.2f}"
                                                                    for c in categories) + " |")
    domains = sorted({r["domain"] for r in test})
    lines += ["", "## By domain (dev-tuned threshold, test)", "",
              "harmful flagged / harmful · non-harmful flagged / non-harmful", "",
              "| model | config | " + " | ".join(domains) + " |", "|---|---|" + "---|" * len(domains)]
    for r in results:
        bd = r["tuned"]["by_domain"]
        lines.append(f"| {r['model']} | {r['config']} | " + " | ".join(
            (f"{bd[d][0]}/{bd[d][1]} · " if bd[d][1] else "— · ") + f"{bd[d][2]}/{bd[d][3]}" for d in domains) + " |")
    (DATA / "results.md").write_text("\n".join(lines) + "\n")

    with open(DATA / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "config", "threshold", "recall", "benign_fp", "hard_fp", "auc", "pairs_ok", "pairs",
                    "latency_ms", "recall_at_0.5", "benign_fp_at_0.5", "hard_fp_at_0.5"])
        for r in results:
            m, f5 = r["tuned"], r["at_05"]
            w.writerow([r["model"], r["config"], round(r["threshold"], 4), round(m["recall"], 4),
                        round(m["benign_fp"], 4), round(m["hard_fp"], 4), round(m["auc"], 4), *m["pairs"],
                        round(m["latency"], 1), round(f5["recall"], 4), round(f5["benign_fp"], 4),
                        round(f5["hard_fp"], 4)])
    print(f"wrote {DATA / 'results.md'} and results.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score uncached prompts")
    r.add_argument("--models", required=True, help="comma-separated: laya, gliner:fp32, gliner:int8, gliner:fp16")
    r.add_argument("--limit", type=int, help="only the first N prompts (smoke test)")
    sub.add_parser("report", help="build results.md / results.csv from the cache")
    args = ap.parse_args()
    if args.cmd == "run":
        run([m.strip() for m in args.models.split(",") if m.strip()], args.limit)
    else:
        report()


if __name__ == "__main__":
    main()
