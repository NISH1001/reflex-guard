"""Build data/eval/ood/prompts.csv from public safety datasets (all CC-BY-4.0), as an
out-of-distribution check: neither model card names them as training data.

    uv run --with pyarrow python scripts/build_ood_data.py

| source | taken | kinds |
|---|---|---|
| nvidia/Aegis-AI-Content-Safety-Dataset-2.0, test split | 200 unsafe prompts with a mapped category, 100 safe | harmful, benign |
| natolambert/xstest-v2-copy (XSTest v2 prompts) | all 450 | 250 hard negatives, 200 harmful |
| bench-llm/or-bench | 120 from or-bench-hard-1k, 120 from or-bench-toxic | hard negatives, harmful |

Sampling is seeded, so the output is reproducible. Aegis rows labeled only "Needs Caution",
and every "Sexual (minor)" row, are left out.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parent.parent / "data" / "eval" / "ood"
SEED = 12
AEGIS = {
    "Criminal Planning/Confessions": "illegal_activity",
    "Illegal Activity": "illegal_activity",
    "Fraud/Deception": "illegal_activity",
    "Hate/Identity Hate": "hate_harassment",
    "Harassment": "hate_harassment",
    "Violence": "violence_weapons",
    "Guns and Illegal Weapons": "violence_weapons",
    "Threat": "violence_weapons",
    "Controlled/Regulated Substances": "dangerous_substances",
    "Sexual": "sexual_content",
    "PII/Privacy": "privacy_surveillance",
    "Suicide and Self Harm": "self_harm",
    "Malware": "cyber_abuse",
}
# XSTest's unsafe contrast prompts, by type; contrast_definitions mixes topics, so no category.
XSTEST = {
    "contrast_homonyms": "violence_weapons",
    "contrast_figurative_language": "violence_weapons",
    "contrast_safe_targets": "violence_weapons",
    "contrast_safe_contexts": "violence_weapons",
    "contrast_historical_events": "violence_weapons",
    "contrast_discr": "hate_harassment",
    "contrast_privacy": "privacy_surveillance",
    "contrast_definitions": "",
}
ORBENCH = {
    "illegal": "illegal_activity", "privacy": "privacy_surveillance", "sexual": "sexual_content",
    "violence": "violence_weapons", "self-harm": "self_harm", "hate": "hate_harassment",
    "harassment": "hate_harassment", "unethical": "", "harmful": "", "deception": "",
}
COLUMNS = ["id", "domain", "kind", "prompt", "categories", "pair_id", "split", "source"]


def row(i: str, domain: str, kind: str, prompt: str, categories: str, source: str) -> dict:
    return {"id": i, "domain": domain, "kind": kind, "prompt": prompt.strip(), "categories": categories,
            "pair_id": "", "split": "test", "source": source}


def aegis(rng: random.Random) -> list[dict]:
    df = pd.read_json(hf_hub_download("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "test.json", repo_type="dataset"))
    df = df[df["prompt"].astype(str).str.strip().ne("") & df["prompt"].ne("REDACTED")]
    unsafe, safe = [], []
    for r in df.itertuples():
        cats = [c.strip() for c in str(r.violated_categories or "").split(",") if c.strip()]
        if "Sexual (minor)" in cats:
            continue
        if r.prompt_label == "unsafe":
            mapped = sorted({AEGIS[c] for c in cats if c in AEGIS})
            if mapped:
                unsafe.append(row(f"aegis-{r.id}", "aegis", "harmful", r.prompt, ";".join(mapped), "Aegis 2.0 test"))
        elif not cats:
            safe.append(row(f"aegis-{r.id}", "aegis", "benign", r.prompt, "", "Aegis 2.0 test"))
    return rng.sample(unsafe, 200) + rng.sample(safe, 100)


def xstest() -> list[dict]:
    df = pd.read_parquet(hf_hub_download("natolambert/xstest-v2-copy", "data/prompts-00000-of-00001.parquet",
                                         repo_type="dataset"))
    out = []
    for r in df.itertuples():
        harmful = r.type.startswith("contrast_")
        out.append(row(f"xstest-{r.id}", "xstest", "harmful" if harmful else "hard_negative", r.prompt,
                       XSTEST[r.type] if harmful else "", f"XSTest v2 ({r.type})"))
    return out


def orbench(rng: random.Random) -> list[dict]:
    out = []
    for name, kind, n in [("or-bench-hard-1k", "hard_negative", 120), ("or-bench-toxic", "harmful", 120)]:
        df = pd.read_csv(hf_hub_download("bench-llm/or-bench", f"{name}.csv", repo_type="dataset"))
        picked = rng.sample(list(df.itertuples()), n)
        out += [row(f"{name}-{r.Index}", "orbench", kind, r.prompt, ORBENCH[r.category] if kind == "harmful" else "",
                    f"OR-Bench {name.removeprefix('or-bench-')} ({r.category})") for r in picked]
    return out


def main() -> None:
    rng = random.Random(SEED)
    rows = aegis(rng) + xstest() + orbench(rng)
    seen, unique = set(), []
    for r in rows:
        if r["prompt"] and r["prompt"] not in seen:
            seen.add(r["prompt"])
            unique.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "prompts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(unique)
    cats = {
        "violence_weapons": "physical harm, threats, attacks, weapons",
        "hate_harassment": "hate speech, harassment or demeaning people for who they are",
        "sexual_content": "explicit sexual content",
        "self_harm": "suicide or self-injury",
        "illegal_activity": "crime, fraud, theft or evading the law",
        "dangerous_substances": "illegal drugs, controlled or toxic substances",
        "privacy_surveillance": "exposing personal data, tracking or identifying people",
        "cyber_abuse": "hacking, malware or unauthorized access",
    }
    (OUT / "categories.json").write_text(json.dumps(cats, indent=2) + "\n")
    print(f"wrote {len(unique)} rows -> {OUT / 'prompts.csv'}")


if __name__ == "__main__":
    main()
