# 04 · Core benchmark: every model, mode and combination

**Dates:** 2026-09-28 → 2026-10-16 · **Version:** reflexguard v0.2.0 · **Data:** `data/eval/prompts.csv` (690 prompts)
· **Full tables:** `data/eval/results.md` · **Figures:** `data/eval/figures/`

## Question

On one fixed, multi-label set, how do the four model families compare in each mode (`noul`, `choice`, `score`) and
combination (`any`, `all`, `votes=2`, `noul|choice`, `noul&choice`)?

## Set and protocol

- **690 prompts**: 238 harmful (akd-guardrails science controls, dual-use twins, public jailbreak / injection
  sets), 185 benign, 267 hard negatives (40 benign twins, NotInject, synthetic risky-sounding rows); 8 categories
  (`violence_weapons`, `cyber_abuse`, `privacy_surveillance`, `illegal_activity`, `sabotage`,
  `dangerous_substances`, `jailbreak`, `prompt_injection`). See `data/eval/README.md`.
- **Split** by hash, twins kept together: dev 368, **test 322** (104 harmful, 94 benign, 124 hard negatives).
- Each model × mode is scored once and cached (`data/eval/cache/<model>.jsonl`). Combinations are computed from the
  cached per-mode scores with reflexguard's own mode expressions; for prompts a guard chunks, modes are combined
  within each chunk first (per-chunk entries). A prompt is flagged when any category's combined score reaches the
  threshold, picked per config on dev (best balanced accuracy) and reported on test.

## Results (test split)

| model | config | threshold | recall | benign FP | hard-neg FP | AUC | twins |
|---|---|---|---|---|---|---|---|
| Laya | noul | 0.360 | 71.2% | 9.6% | 21.0% | 0.857 | 3/18 |
| Laya | all | 0.193 | 70.2% | 9.6% | 17.7% | 0.865 | 6/18 |
| Laya | votes=2 | 0.572 | 57.7% | 3.2% | **7.3%** | 0.835 | 1/18 |
| GLiNER fp32 | score | 0.505 | 72.1% | 11.7% | 18.5% | 0.835 | 13/18 |
| GLiNER fp32 | choice | 0.633 | 54.8% | 5.3% | 8.9% | 0.776 | 8/18 |
| GLiNER int8 | score | 0.531 | 64.4% | 5.3% | 16.9% | 0.840 | 10/18 |
| GLiNER int8 | votes=2 | 0.507 | 76.9% | 13.8% | 36.3% | 0.818 | **15/18** |
| Jev 1.14 | **score** | 0.303 | 76.0% | **1.1%** | 8.1% | **0.940** | 11/18 |
| Jev 1.14 | choice | 0.130 | **84.6%** | 7.4% | 25.8% | 0.910 | 12/18 |
| Jev 1.14 | all | 0.280 | 76.9% | 1.1% | 8.9% | 0.930 | 11/18 |
| GPT-6 Luna | **score** | 0.197 | 79.8% | 2.1% | 8.1% | 0.920 | 13/18 |
| GPT-6 Luna | votes=2 | 0.213 | 81.7% | 4.3% | 12.9% | 0.903 | 12/18 |
| GPT-6 Luna | noul | 0.710 | 76.9% | 4.3% | 12.1% | 0.881 | 10/18 |

GLiNER rows use the default `noul_orders=3`. All 8 configs per model, the fixed-0.5 view, per-category AUC and
per-domain counts are in `results.md`.

## Findings

- **API models lead in every config.** AUC: Jev 0.91–0.94, GPT-6 Luna 0.88–0.92, Laya 0.80–0.87, GLiNER 0.66–0.84.
- **`score` is the best single mode for both API models** (Jev 76% recall at 1.1% / 8.1% FP); Laya does best on
  `noul` / `all`; GLiNER's `noul` is its weakest mode even after the label-order fix (see 05).
- **Combinations help when every mode is sound** (Laya `votes=2`: fewest hard-negative FP of any model), and inherit
  a weak mode's errors when one isn't (GLiNER `any`).
- **Intent:** GLiNER int8 `votes=2` tells apart 15/18 harmful/benign twins; the API models 10–13; Laya ≤ 6.

## Caveats

Content categories have 3–8 labeled test rows each (per-category numbers are indicative); harmful rows are mostly
attacks. Latency columns were recorded under machine load and across different hardware (see 06).
