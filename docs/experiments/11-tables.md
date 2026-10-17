## Single models (each at its dev-picked config)

| guard | core recall | core benign FP | core hard-neg FP | twins | OOD recall | OOD benign FP | OOD hard-neg FP | median latency |
|---|---|---|---|---|---|---|---|---|
| Laya | 65.4% | 6.4% | 12.9% | 4/18 | 59.2% | 35.4% | 31.1% | 198 ms |
| GLiNER | 72.1% | 11.7% | 18.5% | 13/18 | 83.4% | 42.4% | 55.1% | 434 ms |
| Jev | 76.0% | 1.1% | 8.1% | 11/18 | 90.2% | 24.2% | 49.5% | 166 ms |
| GPT-6 Luna | 79.8% | 2.1% | 8.1% | 13/18 | 77.8% | 15.2% | 25.7% | 317 ms |

## Ensembles

| guard | core recall | core benign FP | core hard-neg FP | twins | OOD recall | OOD benign FP | OOD hard-neg FP | median latency |
|---|---|---|---|---|---|---|---|---|
| Laya \| Jev (AnyOf) | 86.5% | 7.4% | 20.2% | 13/18 | 92.5% | 49.5% | 59.7% | 209 ms |
| Laya & Jev (AllOf) | 51.9% | 0.0% | 0.8% | 1/18 | 54.1% | 8.1% | 18.9% | 209 ms |
| Jev \| GPT-6 Luna (AnyOf) | 82.7% | 3.2% | 12.9% | 13/18 | 91.1% | 27.3% | 51.9% | 329 ms |
| Jev & GPT-6 Luna (AllOf) | 72.1% | 0.0% | 3.2% | 11/18 | 76.5% | 11.1% | 22.7% | 329 ms |
| Votes(2): Laya, GLiNER, Jev | 67.3% | 3.2% | 9.7% | 10/18 | 79.0% | 23.2% | 43.0% | 434 ms |
| Votes(2): Laya, Jev, GPT-6 Luna | 77.9% | 1.1% | 3.2% | 12/18 | 79.6% | 13.1% | 32.7% | 356 ms |
| Votes(3): all four | 64.4% | 1.1% | 2.4% | 8/18 | 71.5% | 12.1% | 22.4% | 464 ms |
| Laya \| GLiNER (local only) | 82.7% | 14.9% | 26.6% | 15/18 | 86.9% | 58.6% | 64.6% | 434 ms |

## Lazy ensembles (`lazy=True`: members in order, stop once settled)

| guard | core recall | core benign FP | core hard-neg FP | twins | OOD recall | OOD benign FP | OOD hard-neg FP | median latency  | called an API (core / OOD) |
|---|---|---|---|---|---|---|---|---|---|
| Votes(2): Laya, GLiNER, GPT-6 Luna, eager | 69.2% | 2.1% | 6.5% | 11/18 | 73.8% | 20.2% | 28.9% | 464 ms |
| Votes(2): Laya, GLiNER, GPT-6 Luna, lazy | 69.2% | 2.1% | 6.5% | 11/18 | 73.8% | 20.2% | 28.9% | 676 ms | 39% / 68% |
| Votes(2): Laya, GLiNER, Jev, eager | 67.3% | 3.2% | 9.7% | 10/18 | 79.0% | 23.2% | 43.0% | 434 ms |
| Votes(2): Laya, GLiNER, Jev, lazy | 67.3% | 3.2% | 9.7% | 10/18 | 79.0% | 23.2% | 43.0% | 676 ms | 39% / 68% |
| (Laya \| GLiNER) & Jev & GPT-6 Luna, eager | 61.5% | 0.0% | 2.4% | 8/18 | 69.4% | 10.1% | 16.8% | 464 ms |
| (Laya \| GLiNER) & Jev & GPT-6 Luna, lazy | 61.5% | 0.0% | 2.4% | 8/18 | 69.4% | 10.1% | 16.8% | 478 ms | 41% / 76% |

## Cascades (`Margin(below, above)` around the first guard's own threshold)

| guard | core recall | core benign FP | core hard-neg FP | twins | OOD recall | OOD benign FP | OOD hard-neg FP | median latency  | escalated (core / OOD) |
|---|---|---|---|---|---|---|---|---|---|
| Laya → Jev, Margin(0.02) | 65.4% | 6.4% | 10.5% | 4/18 | 60.1% | 34.3% | 31.9% | 198 ms | 2% / 4% |
| Laya → Jev, Margin(0.05) | 66.3% | 5.3% | 10.5% | 5/18 | 63.8% | 30.3% | 35.9% | 199 ms | 6% / 10% |
| Laya → Jev, Margin(0.1) | 68.3% | 3.2% | 11.3% | 4/18 | 73.0% | 32.3% | 39.7% | 200 ms | 12% / 24% |
| Laya → Jev, Margin(0.15) | 74.0% | 2.1% | 14.5% | 7/18 | 81.7% | 34.3% | 43.5% | 206 ms | 25% / 39% |
| Laya → Jev, Margin(0.2) | 78.8% | 3.2% | 14.5% | 10/18 | 89.8% | 35.4% | 54.1% | 324 ms | 49% / 60% |
| Laya → Jev, Margin(0.3) | 80.8% | 3.2% | 12.1% | 11/18 | 90.8% | 33.3% | 56.2% | 360 ms | 81% / 73% |
| Laya → Jev, Margin(0.1, 0.02) | 71.2% | 6.4% | 12.1% | 6/18 | 73.4% | 39.4% | 41.1% | 199 ms | 9% / 18% |
| Laya → Jev, Margin(0.2, 0.05) | 81.7% | 6.4% | 16.9% | 12/18 | 90.4% | 42.4% | 56.2% | 314 ms | 46% / 50% |
| Laya → Jev, Margin(0.3, 0.1) | 82.7% | 4.3% | 16.9% | 11/18 | 91.5% | 41.4% | 58.1% | 357 ms | 76% / 62% |
| Laya → GPT-6 Luna, Margin(0.02) | 65.4% | 6.4% | 10.5% | 4/18 | 59.5% | 34.3% | 31.1% | 198 ms | 2% / 4% |
| Laya → GPT-6 Luna, Margin(0.05) | 66.3% | 5.3% | 9.7% | 5/18 | 60.7% | 30.3% | 31.6% | 199 ms | 6% / 10% |
| Laya → GPT-6 Luna, Margin(0.1) | 69.2% | 4.3% | 11.3% | 5/18 | 67.4% | 27.3% | 31.6% | 200 ms | 12% / 24% |
| Laya → GPT-6 Luna, Margin(0.15) | 76.0% | 4.3% | 11.3% | 9/18 | 73.8% | 29.3% | 31.6% | 206 ms | 25% / 39% |
| Laya → GPT-6 Luna, Margin(0.2) | 81.7% | 4.3% | 12.1% | 12/18 | 80.0% | 27.3% | 36.5% | 358 ms | 49% / 60% |
| Laya → GPT-6 Luna, Margin(0.3) | 82.7% | 4.3% | 12.1% | 13/18 | 80.0% | 25.3% | 36.2% | 464 ms | 81% / 73% |
| Laya → GPT-6 Luna, Margin(0.1, 0.02) | 71.2% | 6.4% | 12.1% | 6/18 | 68.8% | 36.4% | 34.9% | 199 ms | 9% / 18% |
| Laya → GPT-6 Luna, Margin(0.2, 0.05) | 83.7% | 6.4% | 14.5% | 13/18 | 81.5% | 36.4% | 41.9% | 338 ms | 46% / 50% |
| Laya → GPT-6 Luna, Margin(0.3, 0.1) | 84.6% | 5.3% | 16.9% | 13/18 | 81.3% | 33.3% | 41.6% | 433 ms | 76% / 62% |
| GLiNER → Jev, Margin(0.02) | 72.1% | 6.4% | 13.7% | 13/18 | 86.3% | 39.4% | 54.1% | 439 ms | 14% / 13% |
| GLiNER → Jev, Margin(0.05) | 73.1% | 2.1% | 10.5% | 12/18 | 88.1% | 36.4% | 50.5% | 473 ms | 34% / 32% |
| GLiNER → Jev, Margin(0.1) | 76.9% | 1.1% | 9.7% | 12/18 | 90.4% | 27.3% | 49.5% | 605 ms | 84% / 64% |
| GLiNER → Jev, Margin(0.15) | 76.0% | 1.1% | 8.1% | 11/18 | 90.4% | 26.3% | 50.3% | 614 ms | 95% / 86% |
| GLiNER → Jev, Margin(0.2) | 76.0% | 1.1% | 8.1% | 11/18 | 90.4% | 24.2% | 49.5% | 614 ms | 98% / 96% |
| GLiNER → Jev, Margin(0.3) | 76.0% | 1.1% | 8.1% | 11/18 | 90.2% | 24.2% | 49.5% | 614 ms | 100% / 100% |
| GLiNER → Jev, Margin(0.1, 0.02) | 81.7% | 6.4% | 14.5% | 14/18 | 94.2% | 43.4% | 62.4% | 592 ms | 69% / 33% |
| GLiNER → Jev, Margin(0.2, 0.05) | 78.8% | 2.1% | 10.5% | 13/18 | 92.7% | 38.4% | 58.4% | 603 ms | 80% / 49% |
| GLiNER → Jev, Margin(0.3, 0.1) | 76.9% | 1.1% | 9.7% | 12/18 | 90.9% | 28.3% | 52.4% | 606 ms | 88% / 69% |
| GLiNER → Laya → Jev, [Margin(0.05, 0.05), Margin(0.2, 0.2)] | 76.0% | 3.2% | 13.7% | 12/18 | 87.9% | 39.4% | 53.0% | 473 ms | 34% / 32% |
| GLiNER → Laya → Jev, [Margin(0.05, 0.02), Margin(0.3, 0.1)] | 78.8% | 7.4% | 17.7% | 13/18 | 90.6% | 45.5% | 58.9% | 458 ms | 27% / 21% |
