# reflexguard eval · core set

Test split: 322 of 690 prompts (104 harmful, 94 benign, 124 hard negatives). Thresholds are picked on the dev split (best balanced accuracy) and applied to test. A prompt is flagged when any category's combined score reaches the threshold. Latency is the median per prompt, summed over the modes a config needs.

## Dev-tuned threshold

| model | config | threshold | recall | benign FP | hard-negative FP | AUC | pairs told apart | latency |
|---|---|---|---|---|---|---|---|---|
| gliner:fp32 | noul | 0.695 | 45.2% | 33.0% | 41.9% | 0.554 | 3/18 | 230 ms |
| gliner:fp32 | choice | 0.628 | 56.7% | 6.4% | 9.7% | 0.775 | 8/18 | 576 ms |
| gliner:fp32 | score | 0.505 | 72.1% | 11.7% | 18.5% | 0.835 | 13/18 | 434 ms |
| gliner:fp32 | any | 0.695 | 64.4% | 34.0% | 42.7% | 0.672 | 3/18 | 1240 ms |
| gliner:fp32 | all | 0.393 | 77.9% | 35.1% | 37.1% | 0.795 | 7/18 | 1240 ms |
| gliner:fp32 | votes=2 | 0.486 | 76.0% | 23.4% | 29.8% | 0.793 | 12/18 | 1240 ms |
| gliner:fp32 | noul|choice | 0.695 | 63.5% | 34.0% | 42.7% | 0.670 | 3/18 | 807 ms |
| gliner:fp32 | noul&choice | 0.400 | 79.8% | 35.1% | 35.5% | 0.766 | 8/18 | 807 ms |
| gliner:fp32:orders=2 | noul | 0.557 | 60.6% | 37.2% | 50.0% | 0.625 | 4/18 | 470 ms |
| gliner:fp32:orders=2 | choice | 0.628 | 56.7% | 6.4% | 9.7% | 0.775 | 8/18 | 576 ms |
| gliner:fp32:orders=2 | score | 0.505 | 72.1% | 11.7% | 18.5% | 0.835 | 13/18 | 434 ms |
| gliner:fp32:orders=2 | any | 0.645 | 66.3% | 17.0% | 27.4% | 0.751 | 7/18 | 1486 ms |
| gliner:fp32:orders=2 | all | 0.424 | 74.0% | 21.3% | 29.0% | 0.810 | 6/18 | 1486 ms |
| gliner:fp32:orders=2 | votes=2 | 0.522 | 68.3% | 16.0% | 18.5% | 0.792 | 11/18 | 1486 ms |
| gliner:fp32:orders=2 | noul|choice | 0.645 | 66.3% | 17.0% | 27.4% | 0.750 | 7/18 | 1050 ms |
| gliner:fp32:orders=2 | noul&choice | 0.424 | 75.0% | 23.4% | 29.0% | 0.777 | 6/18 | 1050 ms |
| gliner:fp32:orders=3 | noul | 0.521 | 66.3% | 23.4% | 38.7% | 0.677 | 4/18 | 1460 ms |
| gliner:fp32:orders=3 | choice | 0.628 | 56.7% | 6.4% | 9.7% | 0.775 | 8/18 | 576 ms |
| gliner:fp32:orders=3 | score | 0.505 | 72.1% | 11.7% | 18.5% | 0.835 | 13/18 | 434 ms |
| gliner:fp32:orders=3 | any | 0.628 | 63.5% | 14.9% | 23.4% | 0.760 | 10/18 | 2466 ms |
| gliner:fp32:orders=3 | all | 0.463 | 65.4% | 14.9% | 14.5% | 0.803 | 11/18 | 2466 ms |
| gliner:fp32:orders=3 | votes=2 | 0.514 | 71.2% | 18.1% | 23.4% | 0.787 | 10/18 | 2466 ms |
| gliner:fp32:orders=3 | noul|choice | 0.628 | 63.5% | 14.9% | 23.4% | 0.757 | 10/18 | 2037 ms |
| gliner:fp32:orders=3 | noul&choice | 0.416 | 76.9% | 30.9% | 34.7% | 0.770 | 6/18 | 2037 ms |
| gliner:int8 | noul | 0.597 | 36.5% | 34.0% | 46.0% | 0.445 | 2/18 | 113 ms |
| gliner:int8 | choice | 0.582 | 58.7% | 18.1% | 25.0% | 0.767 | 13/18 | 284 ms |
| gliner:int8 | score | 0.531 | 64.4% | 5.3% | 16.9% | 0.839 | 10/18 | 210 ms |
| gliner:int8 | any | 0.613 | 65.4% | 28.7% | 44.4% | 0.684 | 6/18 | 606 ms |
| gliner:int8 | all | 0.493 | 29.8% | 5.3% | 12.1% | 0.607 | 7/18 | 606 ms |
| gliner:int8 | votes=2 | 0.532 | 74.0% | 29.8% | 39.5% | 0.744 | 11/18 | 606 ms |
| gliner:int8 | noul|choice | 0.613 | 64.4% | 28.7% | 44.4% | 0.679 | 6/18 | 396 ms |
| gliner:int8 | noul&choice | 0.510 | 46.2% | 39.4% | 41.1% | 0.542 | 5/18 | 396 ms |
| gliner:int8:orders=3 | noul | 0.395 | 64.4% | 38.3% | 41.9% | 0.662 | 4/18 | 635 ms |
| gliner:int8:orders=3 | choice | 0.582 | 58.7% | 18.1% | 25.0% | 0.767 | 13/18 | 284 ms |
| gliner:int8:orders=3 | score | 0.531 | 64.4% | 5.3% | 16.9% | 0.839 | 10/18 | 210 ms |
| gliner:int8:orders=3 | any | 0.560 | 71.2% | 25.5% | 35.5% | 0.773 | 12/18 | 1132 ms |
| gliner:int8:orders=3 | all | 0.395 | 64.4% | 38.3% | 40.3% | 0.683 | 4/18 | 1132 ms |
| gliner:int8:orders=3 | votes=2 | 0.507 | 76.9% | 13.8% | 36.3% | 0.817 | 15/18 | 1132 ms |
| gliner:int8:orders=3 | noul|choice | 0.582 | 58.7% | 18.1% | 26.6% | 0.757 | 13/18 | 926 ms |
| gliner:int8:orders=3 | noul&choice | 0.395 | 64.4% | 38.3% | 40.3% | 0.670 | 4/18 | 926 ms |
| laya | noul | 0.360 | 71.2% | 9.6% | 21.0% | 0.858 | 3/18 | 85 ms |
| laya | choice | 0.425 | 64.4% | 10.6% | 23.4% | 0.818 | 2/18 | 112 ms |
| laya | score | 0.552 | 63.5% | 9.6% | 15.3% | 0.800 | 3/18 | 95 ms |
| laya | any | 0.803 | 63.5% | 4.3% | 10.5% | 0.840 | 0/18 | 293 ms |
| laya | all | 0.193 | 70.2% | 8.5% | 17.7% | 0.866 | 6/18 | 293 ms |
| laya | votes=2 | 0.572 | 57.7% | 4.3% | 7.3% | 0.835 | 1/18 | 293 ms |
| laya | noul|choice | 0.803 | 63.5% | 4.3% | 9.7% | 0.840 | 0/18 | 198 ms |
| laya | noul&choice | 0.268 | 65.4% | 6.4% | 12.9% | 0.870 | 4/18 | 198 ms |

## Fixed threshold 0.5

| model | config | recall | benign FP | hard-negative FP | pairs told apart |
|---|---|---|---|---|---|
| gliner:fp32 | noul | 87.5% | 76.6% | 82.3% | 2/18 |
| gliner:fp32 | choice | 79.8% | 42.6% | 36.3% | 5/18 |
| gliner:fp32 | score | 74.0% | 13.8% | 20.2% | 14/18 |
| gliner:fp32 | any | 95.2% | 84.0% | 87.9% | 0/18 |
| gliner:fp32 | all | 45.2% | 5.3% | 11.3% | 6/18 |
| gliner:fp32 | votes=2 | 72.1% | 20.2% | 26.6% | 11/18 |
| gliner:fp32 | noul|choice | 94.2% | 83.0% | 87.9% | 0/18 |
| gliner:fp32 | noul&choice | 51.9% | 14.9% | 19.4% | 7/18 |
| gliner:fp32:orders=2 | noul | 74.0% | 48.9% | 62.9% | 6/18 |
| gliner:fp32:orders=2 | choice | 79.8% | 42.6% | 36.3% | 5/18 |
| gliner:fp32:orders=2 | score | 74.0% | 13.8% | 20.2% | 14/18 |
| gliner:fp32:orders=2 | any | 89.4% | 63.8% | 72.6% | 1/18 |
| gliner:fp32:orders=2 | all | 50.0% | 5.3% | 11.3% | 6/18 |
| gliner:fp32:orders=2 | votes=2 | 73.1% | 20.2% | 26.6% | 11/18 |
| gliner:fp32:orders=2 | noul|choice | 88.5% | 62.8% | 72.6% | 1/18 |
| gliner:fp32:orders=2 | noul&choice | 56.7% | 14.9% | 19.4% | 7/18 |
| gliner:fp32:orders=3 | noul | 74.0% | 31.9% | 41.9% | 4/18 |
| gliner:fp32:orders=3 | choice | 79.8% | 42.6% | 36.3% | 5/18 |
| gliner:fp32:orders=3 | score | 74.0% | 13.8% | 20.2% | 14/18 |
| gliner:fp32:orders=3 | any | 87.5% | 55.3% | 55.6% | 1/18 |
| gliner:fp32:orders=3 | all | 53.8% | 6.4% | 10.5% | 8/18 |
| gliner:fp32:orders=3 | votes=2 | 75.0% | 21.3% | 27.4% | 11/18 |
| gliner:fp32:orders=3 | noul|choice | 86.5% | 54.3% | 54.8% | 1/18 |
| gliner:fp32:orders=3 | noul&choice | 59.6% | 16.0% | 20.2% | 6/18 |
| gliner:int8 | noul | 59.6% | 69.1% | 71.0% | 0/18 |
| gliner:int8 | choice | 92.3% | 63.8% | 71.0% | 4/18 |
| gliner:int8 | score | 88.5% | 39.4% | 54.8% | 11/18 |
| gliner:int8 | any | 97.1% | 85.1% | 91.9% | 0/18 |
| gliner:int8 | all | 23.1% | 4.3% | 9.7% | 6/18 |
| gliner:int8 | votes=2 | 89.4% | 46.8% | 59.7% | 10/18 |
| gliner:int8 | noul|choice | 96.2% | 83.0% | 88.7% | 0/18 |
| gliner:int8 | noul&choice | 51.0% | 44.7% | 45.2% | 5/18 |
| gliner:int8:orders=3 | noul | 19.2% | 8.5% | 16.9% | 3/18 |
| gliner:int8:orders=3 | choice | 92.3% | 63.8% | 71.0% | 4/18 |
| gliner:int8:orders=3 | score | 88.5% | 39.4% | 54.8% | 11/18 |
| gliner:int8:orders=3 | any | 94.2% | 68.1% | 79.8% | 2/18 |
| gliner:int8:orders=3 | all | 18.3% | 3.2% | 7.3% | 3/18 |
| gliner:int8:orders=3 | votes=2 | 82.7% | 20.2% | 39.5% | 15/18 |
| gliner:int8:orders=3 | noul|choice | 92.3% | 63.8% | 72.6% | 3/18 |
| gliner:int8:orders=3 | noul&choice | 19.2% | 7.4% | 14.5% | 4/18 |
| laya | noul | 67.3% | 8.5% | 15.3% | 2/18 |
| laya | choice | 61.5% | 9.6% | 21.0% | 2/18 |
| laya | score | 72.1% | 17.0% | 24.2% | 3/18 |
| laya | any | 80.8% | 24.5% | 40.3% | 4/18 |
| laya | all | 51.0% | 2.1% | 3.2% | 1/18 |
| laya | votes=2 | 63.5% | 6.4% | 11.3% | 1/18 |
| laya | noul|choice | 72.1% | 16.0% | 29.8% | 3/18 |
| laya | noul&choice | 55.8% | 2.1% | 5.6% | 1/18 |

## Per-category AUC (test)

Each category's own score: rows labeled with it vs every non-harmful row. `n` is the number of test rows labeled with the category.

| model | config | violence_weapons (n=5) | cyber_abuse (n=5) | privacy_surveillance (n=3) | illegal_activity (n=8) | sabotage (n=5) | dangerous_substances (n=0) | jailbreak (n=32) | prompt_injection (n=48) |
|---|---|---|---|---|---|---|---|---|---|
| gliner:fp32 | noul | 0.70 | 0.98 | 0.98 | 0.79 | 0.98 | — | 0.97 | 0.89 |
| gliner:fp32 | choice | 0.82 | 0.99 | 0.93 | 0.87 | 0.99 | — | 1.00 | 0.86 |
| gliner:fp32 | score | 0.92 | 1.00 | 1.00 | 0.89 | 0.98 | — | 1.00 | 0.78 |
| gliner:fp32 | any | 0.70 | 0.99 | 0.97 | 0.89 | 0.99 | — | 0.99 | 0.79 |
| gliner:fp32 | all | 0.83 | 0.98 | 1.00 | 0.79 | 0.99 | — | 0.98 | 0.88 |
| gliner:fp32 | votes=2 | 0.90 | 1.00 | 0.98 | 0.87 | 0.99 | — | 1.00 | 0.86 |
| gliner:fp32 | noul|choice | 0.70 | 0.99 | 0.95 | 0.87 | 0.99 | — | 0.99 | 0.86 |
| gliner:fp32 | noul&choice | 0.82 | 0.98 | 0.98 | 0.79 | 0.99 | — | 0.98 | 0.88 |
| gliner:fp32:orders=2 | noul | 0.77 | 0.98 | 0.98 | 0.80 | 0.98 | — | 0.97 | 0.87 |
| gliner:fp32:orders=2 | choice | 0.82 | 0.99 | 0.93 | 0.87 | 0.99 | — | 1.00 | 0.86 |
| gliner:fp32:orders=2 | score | 0.92 | 1.00 | 1.00 | 0.89 | 0.98 | — | 1.00 | 0.78 |
| gliner:fp32:orders=2 | any | 0.77 | 0.99 | 0.97 | 0.89 | 0.99 | — | 0.99 | 0.80 |
| gliner:fp32:orders=2 | all | 0.84 | 0.98 | 0.99 | 0.80 | 0.99 | — | 0.99 | 0.86 |
| gliner:fp32:orders=2 | votes=2 | 0.90 | 1.00 | 0.98 | 0.87 | 0.99 | — | 1.00 | 0.86 |
| gliner:fp32:orders=2 | noul|choice | 0.77 | 0.99 | 0.94 | 0.87 | 0.99 | — | 0.99 | 0.86 |
| gliner:fp32:orders=2 | noul&choice | 0.84 | 0.98 | 0.97 | 0.80 | 0.99 | — | 0.99 | 0.86 |
| gliner:fp32:orders=3 | noul | 0.83 | 0.97 | 0.98 | 0.80 | 0.98 | — | 0.97 | 0.88 |
| gliner:fp32:orders=3 | choice | 0.82 | 0.99 | 0.93 | 0.87 | 0.99 | — | 1.00 | 0.86 |
| gliner:fp32:orders=3 | score | 0.92 | 1.00 | 1.00 | 0.89 | 0.98 | — | 1.00 | 0.78 |
| gliner:fp32:orders=3 | any | 0.85 | 0.99 | 0.97 | 0.89 | 0.99 | — | 0.99 | 0.80 |
| gliner:fp32:orders=3 | all | 0.85 | 0.97 | 1.00 | 0.80 | 0.99 | — | 0.99 | 0.87 |
| gliner:fp32:orders=3 | votes=2 | 0.85 | 1.00 | 0.97 | 0.89 | 0.99 | — | 1.00 | 0.87 |
| gliner:fp32:orders=3 | noul|choice | 0.82 | 0.99 | 0.95 | 0.89 | 0.99 | — | 0.99 | 0.87 |
| gliner:fp32:orders=3 | noul&choice | 0.85 | 0.97 | 0.97 | 0.80 | 0.99 | — | 0.99 | 0.87 |
| gliner:int8 | noul | 0.77 | 0.97 | 0.99 | 0.85 | 0.97 | — | 0.95 | 0.89 |
| gliner:int8 | choice | 0.88 | 0.99 | 0.94 | 0.88 | 0.98 | — | 0.98 | 0.85 |
| gliner:int8 | score | 0.96 | 0.98 | 0.99 | 0.91 | 0.98 | — | 0.95 | 0.79 |
| gliner:int8 | any | 0.81 | 0.99 | 0.96 | 0.90 | 0.98 | — | 0.96 | 0.82 |
| gliner:int8 | all | 0.97 | 0.97 | 0.99 | 0.85 | 0.97 | — | 0.95 | 0.89 |
| gliner:int8 | votes=2 | 0.88 | 0.98 | 0.99 | 0.89 | 0.98 | — | 0.98 | 0.84 |
| gliner:int8 | noul|choice | 0.81 | 0.99 | 0.94 | 0.88 | 0.98 | — | 0.98 | 0.85 |
| gliner:int8 | noul&choice | 0.88 | 0.97 | 0.99 | 0.85 | 0.97 | — | 0.95 | 0.89 |
| gliner:int8:orders=3 | noul | 0.86 | 0.97 | 0.99 | 0.84 | 0.98 | — | 0.94 | 0.88 |
| gliner:int8:orders=3 | choice | 0.88 | 0.99 | 0.94 | 0.88 | 0.98 | — | 0.98 | 0.85 |
| gliner:int8:orders=3 | score | 0.96 | 0.98 | 0.99 | 0.91 | 0.98 | — | 0.95 | 0.79 |
| gliner:int8:orders=3 | any | 0.88 | 0.99 | 0.96 | 0.90 | 0.98 | — | 0.96 | 0.82 |
| gliner:int8:orders=3 | all | 0.88 | 0.97 | 0.99 | 0.84 | 0.98 | — | 0.95 | 0.87 |
| gliner:int8:orders=3 | votes=2 | 0.92 | 0.98 | 0.99 | 0.89 | 0.98 | — | 0.98 | 0.84 |
| gliner:int8:orders=3 | noul|choice | 0.88 | 0.99 | 0.94 | 0.88 | 0.98 | — | 0.97 | 0.85 |
| gliner:int8:orders=3 | noul&choice | 0.87 | 0.97 | 0.99 | 0.84 | 0.98 | — | 0.95 | 0.87 |
| laya | noul | 0.93 | 0.97 | 0.95 | 0.82 | 0.97 | — | 0.99 | 0.90 |
| laya | choice | 0.93 | 1.00 | 0.57 | 0.85 | 0.94 | — | 1.00 | 0.88 |
| laya | score | 0.78 | 0.92 | 0.86 | 0.37 | 0.85 | — | 0.98 | 0.78 |
| laya | any | 0.82 | 0.94 | 0.81 | 0.57 | 0.84 | — | 0.99 | 0.86 |
| laya | all | 0.95 | 0.99 | 0.95 | 0.82 | 0.97 | — | 1.00 | 0.89 |
| laya | votes=2 | 0.93 | 0.98 | 0.60 | 0.77 | 0.96 | — | 1.00 | 0.92 |
| laya | noul|choice | 0.92 | 0.98 | 0.58 | 0.84 | 0.95 | — | 0.99 | 0.92 |
| laya | noul&choice | 0.95 | 0.99 | 0.95 | 0.83 | 0.97 | — | 1.00 | 0.89 |

## By domain (dev-tuned threshold, test)

harmful flagged / harmful · non-harmful flagged / non-harmful

| model | config | adversarial | biomedical | chemistry | finance_legal | general | science | technology |
|---|---|---|---|---|---|---|---|---|
| gliner:fp32 | noul | 39/80 · 39/94 | — · 2/10 | — · 3/12 | — · 0/10 | — · 2/13 | 8/24 · 30/66 | — · 7/13 |
| gliner:fp32 | choice | 46/80 · 3/94 | — · 2/10 | — · 3/12 | — · 1/10 | — · 0/13 | 13/24 · 7/66 | — · 2/13 |
| gliner:fp32 | score | 57/80 · 11/94 | — · 3/10 | — · 6/12 | — · 1/10 | — · 0/13 | 18/24 · 10/66 | — · 3/13 |
| gliner:fp32 | any | 56/80 · 40/94 | — · 2/10 | — · 3/12 | — · 0/10 | — · 2/13 | 11/24 · 31/66 | — · 7/13 |
| gliner:fp32 | all | 62/80 · 24/94 | — · 5/10 | — · 6/12 | — · 1/10 | — · 2/13 | 19/24 · 33/66 | — · 8/13 |
| gliner:fp32 | votes=2 | 59/80 · 18/94 | — · 3/10 | — · 7/12 | — · 2/10 | — · 1/13 | 20/24 · 23/66 | — · 5/13 |
| gliner:fp32 | noul|choice | 55/80 · 40/94 | — · 2/10 | — · 3/12 | — · 0/10 | — · 2/13 | 11/24 · 31/66 | — · 7/13 |
| gliner:fp32 | noul&choice | 63/80 · 23/94 | — · 4/10 | — · 6/12 | — · 1/10 | — · 2/13 | 20/24 · 34/66 | — · 7/13 |
| gliner:fp32:orders=2 | noul | 49/80 · 42/94 | — · 2/10 | — · 7/12 | — · 0/10 | — · 3/13 | 14/24 · 35/66 | — · 8/13 |
| gliner:fp32:orders=2 | choice | 46/80 · 3/94 | — · 2/10 | — · 3/12 | — · 1/10 | — · 0/13 | 13/24 · 7/66 | — · 2/13 |
| gliner:fp32:orders=2 | score | 57/80 · 11/94 | — · 3/10 | — · 6/12 | — · 1/10 | — · 0/13 | 18/24 · 10/66 | — · 3/13 |
| gliner:fp32:orders=2 | any | 53/80 · 16/94 | — · 2/10 | — · 6/12 | — · 1/10 | — · 2/13 | 16/24 · 20/66 | — · 3/13 |
| gliner:fp32:orders=2 | all | 60/80 · 15/94 | — · 3/10 | — · 6/12 | — · 1/10 | — · 1/13 | 17/24 · 24/66 | — · 6/13 |
| gliner:fp32:orders=2 | votes=2 | 53/80 · 8/94 | — · 3/10 | — · 5/12 | — · 1/10 | — · 0/13 | 18/24 · 18/66 | — · 3/13 |
| gliner:fp32:orders=2 | noul|choice | 53/80 · 16/94 | — · 2/10 | — · 6/12 | — · 1/10 | — · 2/13 | 16/24 · 20/66 | — · 3/13 |
| gliner:fp32:orders=2 | noul&choice | 61/80 · 16/94 | — · 3/10 | — · 6/12 | — · 1/10 | — · 1/13 | 17/24 · 25/66 | — · 6/13 |
| gliner:fp32:orders=3 | noul | 52/80 · 26/94 | — · 2/10 | — · 7/12 | — · 1/10 | — · 4/13 | 17/24 · 25/66 | — · 5/13 |
| gliner:fp32:orders=3 | choice | 46/80 · 3/94 | — · 2/10 | — · 3/12 | — · 1/10 | — · 0/13 | 13/24 · 7/66 | — · 2/13 |
| gliner:fp32:orders=3 | score | 57/80 · 11/94 | — · 3/10 | — · 6/12 | — · 1/10 | — · 0/13 | 18/24 · 10/66 | — · 3/13 |
| gliner:fp32:orders=3 | any | 47/80 · 13/94 | — · 2/10 | — · 7/12 | — · 1/10 | — · 2/13 | 19/24 · 16/66 | — · 2/13 |
| gliner:fp32:orders=3 | all | 51/80 · 8/94 | — · 2/10 | — · 5/12 | — · 1/10 | — · 0/13 | 17/24 · 13/66 | — · 3/13 |
| gliner:fp32:orders=3 | votes=2 | 56/80 · 11/94 | — · 3/10 | — · 6/12 | — · 1/10 | — · 1/13 | 18/24 · 20/66 | — · 4/13 |
| gliner:fp32:orders=3 | noul|choice | 47/80 · 13/94 | — · 2/10 | — · 7/12 | — · 1/10 | — · 2/13 | 19/24 · 16/66 | — · 2/13 |
| gliner:fp32:orders=3 | noul&choice | 62/80 · 20/94 | — · 4/10 | — · 6/12 | — · 2/10 | — · 1/13 | 18/24 · 33/66 | — · 6/13 |
| gliner:int8 | noul | 27/80 · 32/94 | — · 3/10 | — · 6/12 | — · 2/10 | — · 5/13 | 11/24 · 35/66 | — · 6/13 |
| gliner:int8 | choice | 41/80 · 11/94 | — · 3/10 | — · 8/12 | — · 2/10 | — · 3/13 | 20/24 · 15/66 | — · 6/13 |
| gliner:int8 | score | 53/80 · 9/94 | — · 2/10 | — · 5/12 | — · 2/10 | — · 0/13 | 14/24 · 5/66 | — · 3/13 |
| gliner:int8 | any | 51/80 · 27/94 | — · 5/10 | — · 8/12 | — · 4/10 | — · 3/13 | 17/24 · 30/66 | — · 5/13 |
| gliner:int8 | all | 19/80 · 6/94 | — · 1/10 | — · 4/12 | — · 0/10 | — · 0/13 | 12/24 · 6/66 | — · 3/13 |
| gliner:int8 | votes=2 | 56/80 · 23/94 | — · 5/10 | — · 9/12 | — · 2/10 | — · 3/13 | 21/24 · 28/66 | — · 7/13 |
| gliner:int8 | noul|choice | 50/80 · 27/94 | — · 5/10 | — · 8/12 | — · 4/10 | — · 3/13 | 17/24 · 30/66 | — · 5/13 |
| gliner:int8 | noul&choice | 34/80 · 30/94 | — · 3/10 | — · 6/12 | — · 1/10 | — · 3/13 | 14/24 · 38/66 | — · 7/13 |
| gliner:int8:orders=3 | noul | 52/80 · 27/94 | — · 3/10 | — · 9/12 | — · 2/10 | — · 6/13 | 15/24 · 36/66 | — · 5/13 |
| gliner:int8:orders=3 | choice | 41/80 · 11/94 | — · 3/10 | — · 8/12 | — · 2/10 | — · 3/13 | 20/24 · 15/66 | — · 6/13 |
| gliner:int8:orders=3 | score | 53/80 · 9/94 | — · 2/10 | — · 5/12 | — · 2/10 | — · 0/13 | 14/24 · 5/66 | — · 3/13 |
| gliner:int8:orders=3 | any | 53/80 · 22/94 | — · 3/10 | — · 8/12 | — · 2/10 | — · 3/13 | 21/24 · 24/66 | — · 6/13 |
| gliner:int8:orders=3 | all | 52/80 · 26/94 | — · 3/10 | — · 9/12 | — · 2/10 | — · 5/13 | 15/24 · 36/66 | — · 5/13 |
| gliner:int8:orders=3 | votes=2 | 59/80 · 20/94 | — · 3/10 | — · 8/12 | — · 2/10 | — · 4/13 | 21/24 · 15/66 | — · 6/13 |
| gliner:int8:orders=3 | noul|choice | 41/80 · 13/94 | — · 3/10 | — · 8/12 | — · 2/10 | — · 3/13 | 20/24 · 15/66 | — · 6/13 |
| gliner:int8:orders=3 | noul&choice | 52/80 · 26/94 | — · 3/10 | — · 9/12 | — · 2/10 | — · 5/13 | 15/24 · 36/66 | — · 5/13 |
| laya | noul | 67/80 · 21/94 | — · 1/10 | — · 1/12 | — · 0/10 | — · 2/13 | 7/24 · 4/66 | — · 6/13 |
| laya | choice | 60/80 · 26/94 | — · 1/10 | — · 2/12 | — · 3/10 | — · 2/13 | 7/24 · 3/66 | — · 2/13 |
| laya | score | 60/80 · 18/94 | — · 1/10 | — · 0/12 | — · 1/10 | — · 1/13 | 6/24 · 5/66 | — · 2/13 |
| laya | any | 62/80 · 11/94 | — · 0/10 | — · 0/12 | — · 0/10 | — · 1/13 | 4/24 · 2/66 | — · 3/13 |
| laya | all | 62/80 · 15/94 | — · 1/10 | — · 3/12 | — · 0/10 | — · 2/13 | 11/24 · 5/66 | — · 4/13 |
| laya | votes=2 | 56/80 · 9/94 | — · 0/10 | — · 0/12 | — · 0/10 | — · 2/13 | 4/24 · 0/66 | — · 2/13 |
| laya | noul|choice | 62/80 · 10/94 | — · 0/10 | — · 0/12 | — · 0/10 | — · 1/13 | 4/24 · 2/66 | — · 3/13 |
| laya | noul&choice | 59/80 · 13/94 | — · 1/10 | — · 2/12 | — · 0/10 | — · 2/13 | 9/24 · 1/66 | — · 3/13 |
