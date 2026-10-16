# 06 · GlinerGuard speed: batching, fewer passes, CoreML

**Date:** 2026-09-25 · **Outcome:** none helps on CPU; `batch_size` stays (for GPU providers) but defaults to 1;
int8 is the CPU lever.

## Setup

GLiNER fp32, 8 categories, all three modes in one `GlinerGuard` call, 40 core test prompts, median, laptop CPU
(Apple Silicon), measured side by side with the machine otherwise idle (an earlier run under load average 14–18 was
discarded).

## Results

| setup | median |
|---|---|
| 3 single-mode guards (baseline) | 1,246 ms |
| one all-modes guard, `batch_size=1` | 1,240 ms |
| one all-modes guard, `batch_size=16` (padded batches) | 1,504 ms |
| one pass per mode (`prompt_tokens=448`, longer prompts) | 1,310 ms |
| CoreML provider, fp32 / fp16 | 3,286 / 1,880 ms |
| int8 (earlier runs) | about half of fp32 |

Batched scores match one-at-a-time to 5e-6. onnxruntime already spreads a single pass over every core, so padding
only adds work on CPU; on a GPU provider the batch dimension is otherwise idle and batching pays.

## Note on latency elsewhere in these docs

Latencies recorded during eval runs (`data/eval/cache`) were often taken while other jobs loaded the machine (load
average 10–60). Read them as relative, not absolute, and never across hardware: Laya runs on the laptop GPU (MPS),
GLiNER on CPU, Jev over the network.
