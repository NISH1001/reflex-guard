# 09 · TypeSafe Jev 1.14 on the core and out-of-distribution sets

**Date:** 2026-10-16 · **Model:** `typesafe/jev-1.14-20261014` (pinned; every cache entry records the snapshot that
served it) · **Guard:** `JevGuard` (`reflexguard[jev]`) · **Cost:** $0.165 for 5,034 requests, 3.9M input tokens.

## Question

How does Jev, TypeSafe's System One model served through OpenRouter's Decisions API, compare with Laya and
GLiNER2.5-Decide under the same protocol?

## Protocol (identical to the other models)

- Same prompts (`data/eval/prompts.csv`, `data/eval/ood/prompts.csv`), same 8 categories per set, same questions:
  reflexguard's `noul` / `choice` / `score` questions, sent unchanged — Jev answers these types natively.
- Each mode asked in its own request (as for the other models); every combination computed offline from the cached
  per-mode scores; thresholds tuned on the core dev split (best balanced accuracy) and applied unchanged to the test
  split and to the OOD set.
- Prompts are scored 16 at a time (it is a remote API); latency is timed per call. No prompt exceeded Jev's 30,000-byte
  chunk size, so nothing was chunked.
- Jev returns probabilities rounded to two decimals, which creates ties; AUC averages ranks over ties.

## Results

### Core set (test split, 322 prompts)

| config | threshold | recall | benign FP | hard-neg FP | AUC | twins told apart |
|---|---|---|---|---|---|---|
| noul | 0.41 | 78.8% | 4.3% | 18.5% | 0.910 | 10/18 |
| choice | 0.13 | 84.6% | 7.4% | 25.8% | 0.910 | 12/18 |
| **score** | 0.30 | 76.0% | **1.1%** | **8.1%** | **0.940** | 11/18 |
| any | 0.40 | 79.8% | 4.3% | 19.4% | 0.911 | 10/18 |
| all | 0.28 | 76.9% | 1.1% | 8.9% | 0.930 | 11/18 |
| votes=2 | 0.43 | 76.9% | 3.2% | 15.3% | 0.921 | 10/18 |

### Out-of-distribution set (988 prompts, core-dev thresholds)

| config | recall | benign FP | hard-neg FP | AUC |
|---|---|---|---|---|
| noul | 91.3% | 25.3% | 55.9% | 0.834 |
| choice | 97.1% | 36.4% | 68.6% | 0.848 |
| **score** | 90.2% | 24.2% | 49.5% | **0.881** |
| any | 92.5% | 28.3% | 57.0% | 0.846 |
| all | 90.9% | 24.2% | 50.5% | 0.881 |
| votes=2 | 90.6% | 24.2% | 53.2% | 0.841 |

Per-category AUC (noul) on the OOD set: violence 0.88, hate/harassment 0.94, sexual 0.99, self-harm 0.96, illegal
0.95, dangerous substances 0.99, privacy 0.95 (cyber abuse 0.84, n = 1).

## Findings

- **Best ranking of the three models on both sets.** Core AUC 0.91–0.94 in every mode (Laya 0.80–0.87, GLiNER
  0.68–0.84); OOD 0.83–0.88 (Laya 0.66–0.71, GLiNER 0.68–0.76 in the 0.1.1 run).
- **`score` is its best mode**, unlike the other two: 76% recall at 1.1% benign and 8.1% hard-negative false positives
  on core, the best operating point measured.
- **Its core thresholds do not transfer.** The tuned thresholds are low (0.13–0.43); on the public set they catch
  90–97% of harmful prompts but flag 50–69% of benign look-alikes (XSTest, OR-Bench). The ranking transfers, the cut-off
  does not — the case for per-deployment calibration.
- **Intent:** 10–12 of 18 harmful/benign twins told apart (Laya ≤ 6, GLiNER up to 15).
- **Latency** ~165 ms per request, mostly network; not comparable with the local models (see 06).

## Reproduce

```bash
export OPENROUTER_API_KEY=...
uv run python scripts/eval.py run --models jev && uv run python scripts/eval.py report
uv run python scripts/eval.py --set ood run --models jev && uv run python scripts/eval.py --set ood report
```
