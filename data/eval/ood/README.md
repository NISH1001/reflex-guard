# Out-of-distribution eval set

988 prompts from three public safety datasets, built by `scripts/build_ood_data.py` (seeded, reproducible). Same
columns as `../prompts.csv`; every row is `split=test`. **Nothing is tuned here**: each model × config keeps the
threshold it was given on the core set's dev split, so this set measures how well a guard *transfers*, as well as
threshold-free AUC.

| source | license | taken | kinds |
|---|---|---|---|
| [nvidia/Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) (Jan 2025), test split | CC-BY-4.0 | 200 unsafe prompts with a mapped category, 99 safe | harmful, benign |
| [XSTest v2](https://huggingface.co/datasets/natolambert/xstest-v2-copy) (Röttger et al., 2023) | CC-BY-4.0 | all 449 unique prompts | 250 hard negatives (safe prompts that look unsafe), 199 harmful contrasts |
| [OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench) (Cui et al., 2024) | CC-BY-4.0 | 120 from `or-bench-hard-1k`, 120 from `or-bench-toxic` | hard negatives (over-refusal bait), harmful |

- **"Out of distribution" is as far as the model cards say.** Neither GLiNER2.5-Decide's card nor Laya's README names
  its training data, so these sets are "not named as training data", not "provably unseen". All three are older than
  both models.
- **Categories** (`categories.json`): 8 content categories. Source labels are mapped in `build_ood_data.py`
  (`AEGIS`, `XSTEST`, `ORBENCH`); 57 harmful rows (XSTest `contrast_definitions`, OR-Bench `unethical` / `harmful` /
  `deception`) have no category and count only in the prompt-level metrics. `cyber_abuse` has a single row.
- **Left out:** Aegis rows labeled only "Needs Caution", and every Aegis row labeled "Sexual (minor)".
- Aegis categories describe the prompt–response pair; they are used here as the prompt's categories.
