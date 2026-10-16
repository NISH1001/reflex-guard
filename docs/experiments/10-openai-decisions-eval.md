# 10 · OpenAI Decisions API (GPT-6 Luna) on the core and out-of-distribution sets

**Date:** 2026-10-16 · **Guard:** `OpenAIDecisionGuard` (`reflexguard[openai]`) · **Model:** `gpt-6-luna` via
`POST https://api.openai.com/v1/decisions` (OpenAI's own API; no dated snapshot exists, so cache entries record the
served model and the run date)

## Question

How does GPT-6 Luna, through OpenAI's typed-decision endpoint, compare under the same protocol as the other models?

## Setup

- reflexguard's questions map to OpenAI's types: `noul` → `predicate`, `choice` → `choice`, `score` → `score`
  (shapes from the official `openai` SDK's `decisions` types). Same prompts, categories and threshold protocol as 04 / 07.
- Each mode in its own request, 8 prompts in flight, retries with backoff.
- An earlier partial run through OpenRouter (`openai/gpt-6-luna-decisions-20261006`) was discarded; every number
  here is from OpenAI's API directly.

## Results

| set | config | recall | benign FP | hard-neg FP | AUC | twins |
|---|---|---|---|---|---|---|
| core | noul | 76.9% | 4.3% | 12.1% | 0.881 | 10/18 |
| core | choice | 81.7% | 4.3% | 15.3% | 0.898 | 12/18 |
| core | **score** | 79.8% | 2.1% | 8.1% | **0.920** | 13/18 |
| core | votes=2 | 81.7% | 4.3% | 12.9% | 0.903 | 12/18 |
| ood | noul | 73.8% | **9.1%** | 30.5% | 0.810 | — |
| ood | choice | 85.0% | 18.2% | 43.5% | 0.832 | — |
| ood | **score** | 77.8% | 15.2% | 25.7% | 0.855 | — |

## Refusals

The Decisions API can refuse a single question. `OpenAIDecisionGuard(on_refusal="flag")`, the default, scores a
refused question 1.0 so a refusal never reads as safe. Prompts with at least one refused question:

| set | `noul` (predicate) | `choice` | `score` |
|---|---|---|---|
| core (690) | 0 | 23 | 12 |
| ood (988) | 0 | 231 | 126 |

Refusals fall on harmful prompts (most refused categories: illegal activity, violence, privacy). So its `choice` /
`score` numbers, especially on the OOD set, partly measure the fail-closed policy; `noul` is unaffected. Each cache
entry lists the refused categories, so results can be recomputed under `on_refusal="ignore"`.

## Findings

- Second-best ranking on both sets (core 0.88–0.92, OOD 0.81–0.86), behind Jev.
- **Least over-blocking on public data** (OOD `noul` 9.1% benign FP vs Jev 25.3%, Laya 34.3%).
- Refuses a meaningful share of `choice` / `score` questions on harmful text; use `noul` (predicate) when a model
  score, not a policy decision, is wanted.
- OpenAI's native response reports tokens but no cost: this run used 2.53M input tokens on the core set and 3.39M on
  the OOD set, across the three modes.
