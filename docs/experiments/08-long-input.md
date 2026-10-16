# 08 · Long input: two truncation bugs (0.1.1) and per-chunk merging (0.1.2)

**Dates:** 2026-09-28 (0.1.1), 2026-10-16 (0.1.2)

## Bugs found (0.1.1)

| bug | effect | fix |
|---|---|---|
| `LayaGuard` read only Laya's window (`max_len − head_max_len`, ~300 tokens); Laya truncates silently | a threat after ~800 words of filler scored **0.009** for violence | chunk with Laya's own tokenizer, one `predict_batch` call → **0.947** |
| `GlinerGuard` chunked by words, and cut any single "word" longer than a chunk | 1,500 CJK characters: 490 of 1,502 tokens scored; base64: 490 of 3,001 | split long words into overlapping token pieces → all tokens scored |

Both guards raise `InputTooLong` past `max_chunks` (default 64) instead of scoring a prefix.

## How chunks become one result (0.1.2)

0.1.1 took each mode's highest chunk separately and combined those, so `all` / `votes=k` could count two modes as
agreeing when they peaked on different passages. 0.1.2 combines a category's modes **within each chunk**, then keeps
the highest chunk. Categories still peak independently (multi-label); `CategoryResult.chunk` says which chunk set a
score. GLiNER now chunks the text once, sized for its longest prompt, so every mode scores the same passages; overlap
is counted in tokens for both local guards (Jev: characters, since it exposes no tokenizer).

## Eval impact

- 0.1.1: Laya re-scored on the 40 core and 8 OOD prompts past its window; its results moved by at most one prompt
  (the long prompts are mostly jailbreaks whose opening already gives them away). No eval prompt had a word long
  enough to hit the GLiNER bug.
- 0.1.2: every prompt a guard may chunk is re-scored, and the eval caches per-chunk scores for them so multi-mode
  configs combine per chunk, as the library does. Single-chunk prompts are unchanged (re-scoring a sample reproduced
  the cache to 5e-6).
