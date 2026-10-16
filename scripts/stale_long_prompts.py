"""List (and with --drop, remove) cached eval scores for prompts a guard may split into chunks.

    uv run python scripts/stale_long_prompts.py          # count
    uv run python scripts/stale_long_prompts.py --drop   # remove them, then `eval.py run` re-scores them

Run after a change to how guards chunk long text (0.1.1: Laya chunks at all, GLiNER splits long
words; 0.1.2: chunks overlap by tokens, GLiNER's modes share chunks, modes combine per chunk).
A prompt that fits one chunk is scored the same way throughout, so its cached scores stay.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETS = [ROOT / "data" / "eval", ROOT / "data" / "eval" / "ood"]


def main() -> None:
    from laya import Router

    from reflexguard.impls._gliner_onnx import words
    from reflexguard.impls.gliner import load_runtime

    router = Router()
    agent = router.load(router.route({"text": "x"}, {"q": {"type": "noul", "instructions": "x"}})["model"])
    laya_budget = agent.cfg.get("max_len", 512) - agent.cfg.get("head_max_len", 192) - 16
    rt = load_runtime()
    drop = "--drop" in sys.argv
    for d in SETS:
        rows = {r["id"]: r["prompt"] for r in csv.DictReader(open(d / "prompts.csv"))}
        laya_long = {i for i, p in rows.items()
                     if len(agent.tok(p, add_special_tokens=False, verbose=False)["input_ids"]) > laya_budget}
        # GLiNER's smallest text budget: 512 tokens minus a prompt of up to prompt_tokens (256).
        gliner_long = {i for i, p in rows.items() if sum(len(rt.ids(w)) for w in words(p)) > 256}
        print(f"{d.relative_to(ROOT)}: {len(laya_long)} prompts past Laya's window, "
              f"{len(gliner_long)} past GLiNER's smallest one")
        for path in sorted((d / "cache").glob("*.jsonl")):
            stale = laya_long if path.stem == "laya" else gliner_long
            lines = path.read_text().splitlines()
            keep = [l for l in lines if json.loads(l)["id"] not in stale]
            print(f"  {path.name}: {len(lines) - len(keep)} stale")
            if drop and len(keep) < len(lines):
                path.write_text("".join(l + "\n" for l in keep))


if __name__ == "__main__":
    main()
