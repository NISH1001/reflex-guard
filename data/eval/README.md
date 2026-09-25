# reflexguard eval set

`scripts/eval.py` benchmarks models × modes × combinations on these prompts. See the repo README for how to run it.

| file | what |
|---|---|
| `prompts.csv` | 690 prompts: `id, domain, kind, prompt, categories, pair_id, split, source` |
| `categories.json` | the 8 categories (with descriptions) every model and mode is asked |
| `synthetic.csv` | the synthetic rows, before `scripts/build_eval_data.py` merges them in |
| `cache/<model>.jsonl` | per-mode, per-category scores and latency for every prompt |
| `results.md`, `results.csv` | the latest report |

- **`kind`**: `harmful` (should be flagged), `benign`, or `hard_negative` (benign text built to look risky: trigger words, acquisitive phrasing, the benign twin of a harmful prompt).
- **`categories`**: every category that applies to a harmful row (`;`-separated); empty for the rest.
- **`pair_id`**: links a harmful prompt to its benign twin; twins always share a split.
- **`split`**: `dev` (thresholds are picked here) or `test` (reported), assigned by hash.

## Where the rows come from

| source | rows | kinds |
|---|---|---|
| [akd-guardrails](https://github.com/NASA-IMPACT/akd-guardrails) issue #12 science set | 82 | science benign + 10 harmful controls |
| akd-guardrails 12-query e2e set | 12 (6 duplicates dropped) | science harmful, benign |
| akd-guardrails dual-use pairs | 80 | 40 harmful / benign twins (science) |
| akd-guardrails adversarial set, sampled from `deepset/prompt-injections`, `jackhhao/jailbreak-classification`, `Lakera/gandalf_ignore_instructions`, `leolee99/NotInject`, `xTRam1/safe-guard-prompt-injection` | 380 | 180 attacks, 150 NotInject hard negatives, 50 benign |
| synthetic (this repo) | 137 | benign and hard negatives: general, technology, biomedical, chemistry, finance/legal, science |

Every harmful row comes from the akd corpora; the synthetic rows are all benign. The per-category labels on harmful rows were added here (`LABELS` in `scripts/build_eval_data.py`); attacks are labeled by their source group (`jailbreak`, or `prompt_injection` for injection and Gandalf rows). The public datasets keep their own licenses; see each dataset card.

## Known weaknesses

- **Thin content categories.** Most harmful rows are attacks (111 `prompt_injection`, 71 `jailbreak`); the content categories have 2–14 rows each, all from the science domain. Per-category numbers for them are indicative only.
- **No harmful rows outside science and attacks.** The general / technology / biomedical / chemistry / finance-legal domains measure over-blocking only.
- **Label noise.** Public attack labels are noisy (some `deepset` "injections" are mild probes), and one e2e row (`e2e007`) is borderline.
- **Synthetic rows were written by a model** and read once; they lean toward textbook phrasing.
