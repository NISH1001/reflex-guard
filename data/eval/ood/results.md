# reflexguard eval · ood set

Test split: 988 of 988 prompts (519 harmful, 99 benign, 370 hard negatives). Nothing is tuned on this set: each config keeps the threshold it was given on the core set's dev split. A prompt is flagged when any category's combined score reaches the threshold. Latency is the median per prompt, summed over the modes a config needs.

## Threshold from the core set's dev split

| model | config | threshold | recall | benign FP | hard-negative FP | AUC | pairs told apart | latency |
|---|---|---|---|---|---|---|---|---|
| gliner:fp32 | noul | 0.695 | 60.1% | 26.3% | 35.1% | 0.678 | 0/0 | 478 ms |
| gliner:fp32 | choice | 0.628 | 75.7% | 32.3% | 40.8% | 0.760 | 0/0 | 884 ms |
| gliner:fp32 | score | 0.505 | 83.4% | 42.4% | 55.1% | 0.750 | 0/0 | 668 ms |
| gliner:fp32 | any | 0.695 | 74.4% | 33.3% | 43.0% | 0.735 | 0/0 | 2055 ms |
| gliner:fp32 | all | 0.393 | 93.4% | 50.5% | 73.0% | 0.748 | 0/0 | 2055 ms |
| gliner:fp32 | votes=2 | 0.486 | 91.7% | 49.5% | 70.0% | 0.736 | 0/0 | 2055 ms |
| gliner:fp32 | noul|choice | 0.695 | 74.2% | 33.3% | 43.0% | 0.735 | 0/0 | 1351 ms |
| gliner:fp32 | noul&choice | 0.400 | 93.6% | 50.5% | 76.5% | 0.728 | 0/0 | 1351 ms |
| gliner:fp32:orders=3 | noul | 0.521 | 82.3% | 39.4% | 62.4% | 0.691 | 0/0 | 654 ms |
| gliner:fp32:orders=3 | choice | 0.628 | 75.7% | 32.3% | 40.8% | 0.760 | 0/0 | 884 ms |
| gliner:fp32:orders=3 | score | 0.505 | 83.4% | 42.4% | 55.1% | 0.750 | 0/0 | 668 ms |
| gliner:fp32:orders=3 | any | 0.628 | 82.3% | 37.4% | 54.6% | 0.747 | 0/0 | 2261 ms |
| gliner:fp32:orders=3 | all | 0.463 | 83.0% | 39.4% | 53.5% | 0.746 | 0/0 | 2261 ms |
| gliner:fp32:orders=3 | votes=2 | 0.514 | 87.7% | 41.4% | 61.4% | 0.737 | 0/0 | 2261 ms |
| gliner:fp32:orders=3 | noul|choice | 0.628 | 82.3% | 37.4% | 53.8% | 0.748 | 0/0 | 1571 ms |
| gliner:fp32:orders=3 | noul&choice | 0.416 | 90.6% | 46.5% | 71.9% | 0.726 | 0/0 | 1571 ms |
| gliner:int8 | noul | 0.597 | 46.4% | 13.1% | 24.3% | 0.685 | 0/0 | 104 ms |
| gliner:int8 | choice | 0.582 | 69.7% | 23.2% | 37.3% | 0.750 | 0/0 | 281 ms |
| gliner:int8 | score | 0.531 | 54.9% | 15.2% | 24.6% | 0.738 | 0/0 | 200 ms |
| gliner:int8 | any | 0.613 | 66.9% | 21.2% | 35.1% | 0.740 | 0/0 | 584 ms |
| gliner:int8 | all | 0.493 | 61.5% | 16.2% | 28.6% | 0.718 | 0/0 | 584 ms |
| gliner:int8 | votes=2 | 0.532 | 66.3% | 18.2% | 34.9% | 0.726 | 0/0 | 584 ms |
| gliner:int8 | noul|choice | 0.613 | 66.9% | 21.2% | 35.1% | 0.741 | 0/0 | 385 ms |
| gliner:int8 | noul&choice | 0.510 | 63.4% | 20.2% | 34.3% | 0.703 | 0/0 | 385 ms |
| gliner:int8:orders=3 | noul | 0.395 | 77.5% | 31.3% | 51.9% | 0.699 | 0/0 | 311 ms |
| gliner:int8:orders=3 | choice | 0.582 | 69.7% | 23.2% | 37.3% | 0.750 | 0/0 | 281 ms |
| gliner:int8:orders=3 | score | 0.531 | 54.9% | 15.2% | 24.6% | 0.738 | 0/0 | 200 ms |
| gliner:int8:orders=3 | any | 0.560 | 77.1% | 35.4% | 43.8% | 0.746 | 0/0 | 790 ms |
| gliner:int8:orders=3 | all | 0.395 | 77.5% | 31.3% | 50.8% | 0.720 | 0/0 | 790 ms |
| gliner:int8:orders=3 | votes=2 | 0.507 | 74.0% | 28.3% | 44.3% | 0.731 | 0/0 | 790 ms |
| gliner:int8:orders=3 | noul|choice | 0.582 | 71.7% | 23.2% | 38.9% | 0.746 | 0/0 | 591 ms |
| gliner:int8:orders=3 | noul&choice | 0.395 | 77.5% | 31.3% | 50.8% | 0.707 | 0/0 | 591 ms |
| laya | noul | 0.360 | 58.8% | 34.3% | 32.2% | 0.696 | 0/0 | 88 ms |
| laya | choice | 0.425 | 60.1% | 36.4% | 37.8% | 0.669 | 0/0 | 134 ms |
| laya | score | 0.552 | 40.8% | 24.2% | 23.5% | 0.661 | 0/0 | 111 ms |
| laya | any | 0.803 | 34.3% | 17.2% | 12.4% | 0.663 | 0/0 | 337 ms |
| laya | all | 0.193 | 68.8% | 41.4% | 41.1% | 0.707 | 0/0 | 337 ms |
| laya | votes=2 | 0.572 | 43.4% | 21.2% | 20.3% | 0.692 | 0/0 | 337 ms |
| laya | noul|choice | 0.803 | 33.7% | 17.2% | 11.6% | 0.664 | 0/0 | 223 ms |
| laya | noul&choice | 0.268 | 59.2% | 35.4% | 31.1% | 0.710 | 0/0 | 223 ms |

## Fixed threshold 0.5

| model | config | recall | benign FP | hard-negative FP | pairs told apart |
|---|---|---|---|---|---|
| gliner:fp32 | noul | 90.9% | 68.7% | 75.1% | 0/0 |
| gliner:fp32 | choice | 91.1% | 48.5% | 66.8% | 0/0 |
| gliner:fp32 | score | 83.8% | 43.4% | 56.8% | 0/0 |
| gliner:fp32 | any | 95.4% | 73.7% | 83.2% | 0/0 |
| gliner:fp32 | all | 76.9% | 35.4% | 44.6% | 0/0 |
| gliner:fp32 | votes=2 | 90.4% | 45.5% | 67.0% | 0/0 |
| gliner:fp32 | noul|choice | 95.4% | 72.7% | 82.4% | 0/0 |
| gliner:fp32 | noul&choice | 85.0% | 39.4% | 58.4% | 0/0 |
| gliner:fp32:orders=3 | noul | 83.8% | 40.4% | 64.9% | 0/0 |
| gliner:fp32:orders=3 | choice | 91.1% | 48.5% | 66.8% | 0/0 |
| gliner:fp32:orders=3 | score | 83.8% | 43.4% | 56.8% | 0/0 |
| gliner:fp32:orders=3 | any | 93.6% | 51.5% | 78.6% | 0/0 |
| gliner:fp32:orders=3 | all | 74.4% | 32.3% | 42.2% | 0/0 |
| gliner:fp32:orders=3 | votes=2 | 89.2% | 45.5% | 65.7% | 0/0 |
| gliner:fp32:orders=3 | noul|choice | 93.6% | 50.5% | 76.8% | 0/0 |
| gliner:fp32:orders=3 | noul&choice | 80.7% | 36.4% | 54.1% | 0/0 |
| gliner:int8 | noul | 67.2% | 40.4% | 39.5% | 0/0 |
| gliner:int8 | choice | 90.2% | 58.6% | 69.7% | 0/0 |
| gliner:int8 | score | 76.1% | 32.3% | 45.4% | 0/0 |
| gliner:int8 | any | 92.5% | 72.7% | 74.6% | 0/0 |
| gliner:int8 | all | 59.3% | 14.1% | 25.1% | 0/0 |
| gliner:int8 | votes=2 | 78.2% | 35.4% | 51.6% | 0/0 |
| gliner:int8 | noul|choice | 92.1% | 70.7% | 72.2% | 0/0 |
| gliner:int8 | noul&choice | 65.1% | 22.2% | 36.8% | 0/0 |
| gliner:int8:orders=3 | noul | 60.7% | 15.2% | 31.6% | 0/0 |
| gliner:int8:orders=3 | choice | 90.2% | 58.6% | 69.7% | 0/0 |
| gliner:int8:orders=3 | score | 76.1% | 32.3% | 45.4% | 0/0 |
| gliner:int8:orders=3 | any | 91.3% | 61.6% | 74.1% | 0/0 |
| gliner:int8:orders=3 | all | 55.9% | 12.1% | 21.4% | 0/0 |
| gliner:int8:orders=3 | votes=2 | 77.1% | 30.3% | 49.2% | 0/0 |
| gliner:int8:orders=3 | noul|choice | 90.8% | 58.6% | 71.9% | 0/0 |
| gliner:int8:orders=3 | noul&choice | 59.9% | 15.2% | 29.2% | 0/0 |
| laya | noul | 50.7% | 27.3% | 25.9% | 0/0 |
| laya | choice | 53.8% | 25.3% | 30.3% | 0/0 |
| laya | score | 47.6% | 27.3% | 31.4% | 0/0 |
| laya | any | 64.4% | 37.4% | 44.9% | 0/0 |
| laya | all | 34.9% | 15.2% | 14.1% | 0/0 |
| laya | votes=2 | 49.3% | 23.2% | 25.1% | 0/0 |
| laya | noul|choice | 60.1% | 33.3% | 36.8% | 0/0 |
| laya | noul&choice | 42.2% | 16.2% | 17.0% | 0/0 |

## Per-category AUC (test)

Each category's own score: rows labeled with it vs every non-harmful row. `n` is the number of test rows labeled with the category.

| model | config | violence_weapons (n=180) | hate_harassment (n=115) | sexual_content (n=31) | self_harm (n=20) | illegal_activity (n=118) | dangerous_substances (n=20) | privacy_surveillance (n=55) | cyber_abuse (n=1) |
|---|---|---|---|---|---|---|---|---|---|
| gliner:fp32 | noul | 0.80 | 0.85 | 0.94 | 0.99 | 0.79 | 0.96 | 0.89 | 0.96 |
| gliner:fp32 | choice | 0.86 | 0.87 | 0.96 | 0.98 | 0.87 | 0.97 | 0.88 | 0.95 |
| gliner:fp32 | score | 0.88 | 0.86 | 0.97 | 0.96 | 0.85 | 0.98 | 0.84 | 0.63 |
| gliner:fp32 | any | 0.85 | 0.87 | 0.96 | 0.98 | 0.86 | 0.98 | 0.88 | 0.91 |
| gliner:fp32 | all | 0.84 | 0.85 | 0.94 | 0.98 | 0.80 | 0.98 | 0.90 | 0.96 |
| gliner:fp32 | votes=2 | 0.85 | 0.87 | 0.97 | 0.98 | 0.85 | 0.97 | 0.88 | 0.74 |
| gliner:fp32 | noul|choice | 0.84 | 0.87 | 0.97 | 0.99 | 0.86 | 0.98 | 0.89 | 0.95 |
| gliner:fp32 | noul&choice | 0.83 | 0.85 | 0.94 | 0.99 | 0.80 | 0.97 | 0.89 | 0.96 |
| gliner:fp32:orders=3 | noul | 0.81 | 0.86 | 0.93 | 0.99 | 0.80 | 0.97 | 0.88 | 0.97 |
| gliner:fp32:orders=3 | choice | 0.86 | 0.87 | 0.96 | 0.98 | 0.87 | 0.97 | 0.88 | 0.95 |
| gliner:fp32:orders=3 | score | 0.88 | 0.86 | 0.97 | 0.96 | 0.85 | 0.98 | 0.84 | 0.63 |
| gliner:fp32:orders=3 | any | 0.87 | 0.87 | 0.96 | 0.98 | 0.86 | 0.98 | 0.88 | 0.90 |
| gliner:fp32:orders=3 | all | 0.83 | 0.85 | 0.94 | 0.98 | 0.81 | 0.98 | 0.89 | 0.97 |
| gliner:fp32:orders=3 | votes=2 | 0.85 | 0.88 | 0.97 | 0.98 | 0.85 | 0.97 | 0.88 | 0.74 |
| gliner:fp32:orders=3 | noul|choice | 0.86 | 0.87 | 0.97 | 0.99 | 0.86 | 0.98 | 0.89 | 0.94 |
| gliner:fp32:orders=3 | noul&choice | 0.82 | 0.86 | 0.93 | 0.99 | 0.81 | 0.97 | 0.88 | 0.97 |
| gliner:int8 | noul | 0.81 | 0.88 | 0.95 | 0.99 | 0.81 | 0.97 | 0.88 | 0.98 |
| gliner:int8 | choice | 0.88 | 0.85 | 0.97 | 0.98 | 0.86 | 0.93 | 0.82 | 0.95 |
| gliner:int8 | score | 0.89 | 0.84 | 0.95 | 0.94 | 0.82 | 0.92 | 0.57 | 0.32 |
| gliner:int8 | any | 0.87 | 0.85 | 0.95 | 0.98 | 0.85 | 0.94 | 0.71 | 0.90 |
| gliner:int8 | all | 0.83 | 0.88 | 0.95 | 0.99 | 0.81 | 0.97 | 0.88 | 0.98 |
| gliner:int8 | votes=2 | 0.87 | 0.85 | 0.97 | 0.97 | 0.84 | 0.91 | 0.79 | 0.52 |
| gliner:int8 | noul|choice | 0.87 | 0.85 | 0.97 | 0.98 | 0.86 | 0.93 | 0.82 | 0.95 |
| gliner:int8 | noul&choice | 0.82 | 0.88 | 0.95 | 0.99 | 0.81 | 0.97 | 0.88 | 0.98 |
| gliner:int8:orders=3 | noul | 0.82 | 0.88 | 0.94 | 0.99 | 0.81 | 0.97 | 0.88 | 0.98 |
| gliner:int8:orders=3 | choice | 0.88 | 0.85 | 0.97 | 0.98 | 0.86 | 0.93 | 0.82 | 0.95 |
| gliner:int8:orders=3 | score | 0.89 | 0.84 | 0.95 | 0.94 | 0.82 | 0.92 | 0.57 | 0.32 |
| gliner:int8:orders=3 | any | 0.88 | 0.85 | 0.95 | 0.98 | 0.85 | 0.94 | 0.72 | 0.90 |
| gliner:int8:orders=3 | all | 0.83 | 0.88 | 0.95 | 0.99 | 0.82 | 0.97 | 0.88 | 0.98 |
| gliner:int8:orders=3 | votes=2 | 0.87 | 0.85 | 0.97 | 0.97 | 0.84 | 0.91 | 0.80 | 0.52 |
| gliner:int8:orders=3 | noul|choice | 0.88 | 0.85 | 0.97 | 0.98 | 0.86 | 0.93 | 0.83 | 0.95 |
| gliner:int8:orders=3 | noul&choice | 0.82 | 0.88 | 0.94 | 0.99 | 0.82 | 0.97 | 0.88 | 0.98 |
| laya | noul | 0.86 | 0.85 | 0.95 | 0.95 | 0.88 | 0.96 | 0.84 | 0.98 |
| laya | choice | 0.87 | 0.77 | 0.95 | 0.94 | 0.80 | 0.79 | 0.69 | 0.95 |
| laya | score | 0.82 | 0.79 | 0.87 | 0.91 | 0.73 | 0.80 | 0.75 | 0.99 |
| laya | any | 0.84 | 0.79 | 0.92 | 0.96 | 0.79 | 0.82 | 0.72 | 0.98 |
| laya | all | 0.87 | 0.83 | 0.95 | 0.95 | 0.87 | 0.94 | 0.84 | 0.97 |
| laya | votes=2 | 0.87 | 0.81 | 0.95 | 0.94 | 0.83 | 0.87 | 0.74 | 0.99 |
| laya | noul|choice | 0.87 | 0.81 | 0.95 | 0.94 | 0.83 | 0.86 | 0.70 | 0.98 |
| laya | noul&choice | 0.87 | 0.83 | 0.95 | 0.95 | 0.87 | 0.94 | 0.84 | 0.97 |

## By domain (dev-tuned threshold, test)

harmful flagged / harmful · non-harmful flagged / non-harmful

| model | config | aegis | orbench | xstest |
|---|---|---|---|---|
| gliner:fp32 | noul | 114/200 · 26/99 | 55/120 · 38/120 | 143/199 · 92/250 |
| gliner:fp32 | choice | 156/200 · 32/99 | 78/120 · 44/120 | 159/199 · 107/250 |
| gliner:fp32 | score | 165/200 · 42/99 | 96/120 · 63/120 | 172/199 · 141/250 |
| gliner:fp32 | any | 151/200 · 33/99 | 71/120 · 48/120 | 164/199 · 111/250 |
| gliner:fp32 | all | 187/200 · 50/99 | 104/120 · 74/120 | 194/199 · 196/250 |
| gliner:fp32 | votes=2 | 181/200 · 49/99 | 104/120 · 77/120 | 191/199 · 182/250 |
| gliner:fp32 | noul|choice | 151/200 · 33/99 | 71/120 · 48/120 | 163/199 · 111/250 |
| gliner:fp32 | noul&choice | 187/200 · 50/99 | 105/120 · 80/120 | 194/199 · 203/250 |
| gliner:fp32:orders=3 | noul | 169/200 · 39/99 | 77/120 · 78/120 | 181/199 · 153/250 |
| gliner:fp32:orders=3 | choice | 156/200 · 32/99 | 78/120 · 44/120 | 159/199 · 107/250 |
| gliner:fp32:orders=3 | score | 165/200 · 42/99 | 96/120 · 63/120 | 172/199 · 141/250 |
| gliner:fp32:orders=3 | any | 167/200 · 37/99 | 83/120 · 62/120 | 177/199 · 140/250 |
| gliner:fp32:orders=3 | all | 170/200 · 39/99 | 84/120 · 56/120 | 177/199 · 142/250 |
| gliner:fp32:orders=3 | votes=2 | 172/200 · 41/99 | 97/120 · 68/120 | 186/199 · 159/250 |
| gliner:fp32:orders=3 | noul|choice | 167/200 · 37/99 | 83/120 · 62/120 | 177/199 · 137/250 |
| gliner:fp32:orders=3 | noul&choice | 182/200 · 46/99 | 98/120 · 77/120 | 190/199 · 189/250 |
| gliner:int8 | noul | 82/200 · 13/99 | 48/120 · 28/120 | 111/199 · 62/250 |
| gliner:int8 | choice | 138/200 · 23/99 | 80/120 · 40/120 | 144/199 · 98/250 |
| gliner:int8 | score | 101/200 · 15/99 | 60/120 · 26/120 | 124/199 · 65/250 |
| gliner:int8 | any | 137/200 · 21/99 | 69/120 · 41/120 | 141/199 · 89/250 |
| gliner:int8 | all | 124/200 · 16/99 | 56/120 · 28/120 | 139/199 · 78/250 |
| gliner:int8 | votes=2 | 135/200 · 18/99 | 64/120 · 39/120 | 145/199 · 90/250 |
| gliner:int8 | noul|choice | 137/200 · 21/99 | 69/120 · 41/120 | 141/199 · 89/250 |
| gliner:int8 | noul&choice | 128/200 · 20/99 | 58/120 · 36/120 | 143/199 · 91/250 |
| gliner:int8:orders=3 | noul | 157/200 · 31/99 | 70/120 · 63/120 | 175/199 · 129/250 |
| gliner:int8:orders=3 | choice | 138/200 · 23/99 | 80/120 · 40/120 | 144/199 · 98/250 |
| gliner:int8:orders=3 | score | 101/200 · 15/99 | 60/120 · 26/120 | 124/199 · 65/250 |
| gliner:int8:orders=3 | any | 157/200 · 35/99 | 89/120 · 51/120 | 154/199 · 111/250 |
| gliner:int8:orders=3 | all | 157/200 · 31/99 | 70/120 · 59/120 | 175/199 · 129/250 |
| gliner:int8:orders=3 | votes=2 | 144/200 · 28/99 | 81/120 · 53/120 | 159/199 · 111/250 |
| gliner:int8:orders=3 | noul|choice | 142/200 · 23/99 | 81/120 · 45/120 | 149/199 · 99/250 |
| gliner:int8:orders=3 | noul&choice | 157/200 · 31/99 | 70/120 · 59/120 | 175/199 · 129/250 |
| laya | noul | 130/200 · 34/99 | 52/120 · 39/120 | 123/199 · 80/250 |
| laya | choice | 130/200 · 36/99 | 51/120 · 51/120 | 131/199 · 89/250 |
| laya | score | 82/200 · 24/99 | 30/120 · 26/120 | 100/199 · 61/250 |
| laya | any | 67/200 · 17/99 | 31/120 · 17/120 | 80/199 · 29/250 |
| laya | all | 155/200 · 41/99 | 56/120 · 39/120 | 146/199 · 113/250 |
| laya | votes=2 | 89/200 · 21/99 | 34/120 · 22/120 | 102/199 · 53/250 |
| laya | noul|choice | 65/200 · 17/99 | 31/120 · 17/120 | 79/199 · 26/250 |
| laya | noul&choice | 128/200 · 35/99 | 48/120 · 30/120 | 131/199 · 85/250 |
