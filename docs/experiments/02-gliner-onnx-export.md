# 02 · ONNX export of GLiNER2.5-Decide: parity and speed

**Date:** 2026-09-25 · **Artifact:** [`nishparadox/gliner2.5-decide-onnx`](https://huggingface.co/nishparadox/gliner2.5-decide-onnx)
· **Outcome:** torch-free inference, fp32 exact; `reflexguard[gliner]` needs only onnxruntime, tokenizers, numpy.

## Question

Can GLiNER2.5-Decide's classification path run without torch/transformers (>1 GB of dependencies) at the same
scores?

## Method

- Exported encoder + `[L]`-marker gather + classifier head as one graph:
  `(input_ids, attention_mask, label_positions) -> logits`, one logit per label (opset 17, `torch.onnx.export`).
- fp16 via `onnxruntime.transformers.float16` (fp32 I/O kept; the generic converter mis-handled DeBERTa's `Cast`
  nodes); int8 via `quantize_dynamic` (QInt8 weights).
- Reimplemented gliner2's prompt construction in pure Python (`reflexguard/impls/_gliner_onnx.py`): no `[CLS]/[SEP]`,
  each piece tokenized separately, lower-cased words, trailing `.` added when the text has no end punctuation.
- Parity: 6 texts (incl. empty, a URL, an email) × a 7-task schema (1 multi-label, 3 single-label, 3 ordinal) against
  `gliner2==2.0.0`.

## Results

Input ids and label positions are **identical** to gliner2's processor on every text.

| variant | size | max \|ΔP\| vs torch | NOUL only | 7 tasks |
|---|---|---|---|---|
| gliner2 (torch) | — | — | 148 ms | 376 ms |
| fp32 | 1.75 GB | **0.00000** | 88 ms | 357 ms |
| fp16 | 874 MB | ≤ 0.001 | 175 ms | 436 ms |
| int8 | 643 MB | up to 0.18 | **43 ms** | 179 ms |

CPU, 8 threads, median of 20, end to end. fp16 is slower on CPU (it is for GPU). int8's drift pushes clean text up
toward the threshold (banana bread pii CHOICE 0.29 → 0.47), so fp32 is the default.
