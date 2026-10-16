# 05 · GLiNER's NOUL label-order bias and `noul_orders`

**Dates:** 2026-09-25 (found), 2026-09-25 (fixed, default 3) · final numbers from the v0.1.2 run

## Problem

`GlinerGuard` asks NOUL as one multi-label question over all categories (the best shape in the spike, see 01). On the
core set its prompt-level AUC was only **0.55**, although each category's own AUC was 0.70–0.98. The prompt score is
the highest category score, and one category dominated: `violence_weapons`, **listed first**, had a median score of
**0.63 on benign text** (every other category 0.04–0.21) and set the maximum on 179 of 218 benign test rows.

Moving it to the end of the list (30 benign prompts) dropped its benign median from **0.68 to 0.12**; other categories
placed first stayed low (0.13, 0.26). So the multi-label question is sensitive to label order, worst for this label
in first place.

## Fix

`noul_orders=k` asks the NOUL question k times with the categories rotated by k-ths of the list and averages each
category's score. Rotation keeps the prompt length, so every order scores the same text chunks. Each order costs one
more NOUL pass. Default: **3**.

## Results (final)

Thresholds tuned on the core dev split; core = test split, OOD = public set with those thresholds.

| | core NOUL AUC | core `violence_weapons` AUC | core `any` AUC | OOD NOUL AUC |
|---|---|---|---|---|
| GLiNER fp32 · 1 order | 0.553 | 0.70 | 0.675 | 0.678 |
| GLiNER fp32 · 2 orders | 0.625 | 0.77 | 0.751 | — |
| GLiNER fp32 · 3 orders | **0.677** | **0.83** | **0.761** | 0.691 |
| GLiNER int8 · 1 order | 0.448 | 0.77 | 0.685 | 0.685 |
| GLiNER int8 · 3 orders | **0.662** | **0.86** | **0.773** | 0.699 |
| Laya (reference) | 0.857 | 0.93 | 0.840 | 0.695 |

- Averaging orders lifts NOUL's AUC steadily (0.55 → 0.63 → 0.68 for fp32), and every combination that includes NOUL
  with it. It does not reach Laya's NOUL on the core set.
- On the OOD set the gain is small (0.678 → 0.691): there the first category is not an outlier.
- Benign false-positive rates at the tuned threshold are sensitive to where that threshold lands (fp32 · 3 orders
  `any`: 14.9% in the v0.1.1 run, 24.5% in v0.1.2, at nearly the same AUC). Read AUC for the ranking effect.

Figure: `data/eval/figures/paper_noul_orders.{pdf,png}`.
