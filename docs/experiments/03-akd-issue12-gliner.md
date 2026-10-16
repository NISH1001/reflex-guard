# 03 · GLiNER2.5-Decide in the akd-guardrails input-guard comparison

**Date:** 2026-09-25 · **Write-up:** [akd-guardrails#12 comment](https://github.com/NASA-IMPACT/akd-guardrails/issues/12#issuecomment-5837633455)
· **Code:** akd-guardrails `feature/gliner-input-guard-eval`, `scripts/eval.py` (`gliner:<variant>`).

## Question

How does GLiNER2.5-Decide compare, as an input guard, with the guards already measured in akd-guardrails#12
(gliguard, Granite Guardian 3.2-5B, Shieldstral 3B, Laya) on the same four corpora (science 82, 12-query, dual-use
40 pairs, adversarial 380)?

## Setup

Two families of variants: (1) every Laya variant's questions and thresholds asked of GLiNER, model for model;
(2) reflexguard's `GlinerGuard` as shipped (`rg-*`), on Granite's 7 categories. Only GLiNER was run; other rows are
the committed results, which recompute to the published table exactly.

## Results (selected)

| guard | science FP (of 72) | controls | dual-use pairs | adversarial recall / FP |
|---|---|---|---|---|
| Shieldstral (domain) | 0/72 | 10/10 | 27/40 | 38.3% / 0.0% |
| Granite Guardian | 1/72 | 10/10 | 30/40 | 72.2% / 18.0% |
| gliguard | 12/72 | 10/10 | 26/40 | 63.9% / 9.0% |
| Laya · severity | 1/72 | 3/10 | 10/40 | 40.0% / 6.5% |
| GLiNER · severity | 4/72 | 8/10 | 21/40 | 67.2% / 7.5% |
| GLiNER · rg-vote2 | 4/72 | 8/10 | 25/40 | 73.9% / 17.0% |
| GLiNER · generic policy | 27/72 | 10/10 | **31/40** | 78.3% / 19.5% |

- No GLiNER configuration is a standalone input guard: every one with 10/10 controls blocks ≥ 20.8% of benign science.
- Question for question it beats Laya, most clearly on dual-use twins.
- Long, paragraph-length policy instructions (domain, injection) flag nearly everything.
