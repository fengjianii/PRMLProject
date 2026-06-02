# LightGBM Auto-Tuning Report
Generated: 2026-06-02 17:10:57

## 1. Tuning Summary
- Algorithm: bayesian
- Total Trials: 30
- Completed Trials: 23
- Total Time: 2842.7s (0.79h)
- Best Pearson: 0.0579

## 2. Best Hyperparameters
```json
{
  "max_depth": 5,
  "num_leaves": 27,
  "learning_rate": 0.01815520160405553,
  "n_estimators": 900,
  "subsample": 0.652247529241807,
  "colsample_bytree": 0.6620285060340783,
  "min_child_samples": 250,
  "reg_alpha": 0.2663966663645524,
  "reg_lambda": 0.3503776504888305
}
```

## 3. Performance Improvement
- Baseline Pearson: 0.0576
- Optimized Pearson: 0.0579
- Improvement: +0.0003 (+0.5%)
- Target Pearson: 0.09
- Target Reached: No

## 4. Top 5 Models
| Rank | Pearson | R2 | MSE | Key Params |
|------|---------|-----|-----|------------|
| 1 | 0.0579 | 0.00288 | 0.000026 | depth=5, lr=0.01815520160405553, leaves=27 |
| 2 | 0.0577 | 0.00291 | 0.000026 | depth=5, lr=0.027493458177375345, leaves=28 |
| 3 | 0.0566 | 0.00275 | 0.000026 | depth=5, lr=0.009099831990719233, leaves=28 |
| 4 | 0.0550 | 0.00255 | 0.000026 | depth=3, lr=0.012568672294227057, leaves=7 |
| 5 | 0.0517 | 0.00197 | 0.000026 | depth=3, lr=0.011687206103198827, leaves=7 |

## 5. Parameter Importance
| Parameter | Importance |
|-----------|------------|
| colsample_bytree | 0.3018 |
| reg_lambda | 0.1822 |
| learning_rate | 0.1182 |
| n_estimators | 0.1089 |
| subsample | 0.0926 |
| reg_alpha | 0.0834 |
| max_depth | 0.0670 |
| min_child_samples | 0.0458 |

## 6. Overfitting Analysis
- Best trial overfitting ratio (val_pearson/train_pearson): 0.432
- Overfitting check (ratio >= 0.9): Warning
- Train Pearson: 0.1339
- Val Pearson: 0.0579

## 7. How to Apply Best Params
```python
from mdl import MeowModel
model = MeowModel(cacheDir=None)
model.load_params_from_json('tuning_output/best_params.json')
# or manually:
model.update_params({
    'max_depth': 5,
    'num_leaves': 27,
    'learning_rate': 0.01815520160405553,
    'n_estimators': 900,
    'subsample': 0.652247529241807,
    'colsample_bytree': 0.6620285060340783,
    'min_child_samples': 250,
    'reg_alpha': 0.2663966663645524,
    'reg_lambda': 0.3503776504888305,
})
model.fit(xdf, ydf)
```

## 8. Trial Details
| Trial | Pearson | R2 | MSE | TrainPearson | OverfitRatio | BestIter | Time(s) |
|-------|---------|-----|-----|-------------|-------------|----------|--------|
| 0 | 0.0286 | 0.00091 | 0.000025 | 0.0715 | 0.400 | 131 | 51.9 |
| 1 | 0.0301 | 0.00114 | 0.000028 | 0.0749 | 0.402 | 45 | 17.1 |
| 2 | 0.0383 | 0.00132 | 0.000030 | 0.0561 | 0.683 | 158 | 48.1 |
| 3 | 0.0463 | 0.00134 | 0.000026 | 0.0899 | 0.515 | 700 | 166.6 |
| 4 | 0.0377 | 0.00159 | 0.000027 | 0.0592 | 0.637 | 225 | 61.6 |
| 5 | 0.0428 | -0.00037 | 0.000027 | 0.0950 | 0.451 | 140 | 41.5 |
| 6 | 0.0386 | 0.00090 | 0.000027 | 0.0938 | 0.412 | 300 | 71.2 |
| 8 | 0.0506 | 0.00241 | 0.000026 | 0.0948 | 0.533 | 292 | 73.5 |
| 9 | 0.0457 | 0.00144 | 0.000026 | 0.0752 | 0.608 | 400 | 87.4 |
| 10 | 0.0550 | 0.00255 | 0.000026 | 0.1017 | 0.541 | 881 | 171.6 |
| 11 | 0.0517 | 0.00197 | 0.000026 | 0.1013 | 0.510 | 785 | 138.3 |
| 12 | 0.0428 | 0.00124 | 0.000027 | 0.0892 | 0.479 | 580 | 127.0 |
| 13 | 0.0337 | 0.00142 | 0.000028 | 0.0744 | 0.453 | 439 | 67.6 |
| 14 | 0.0259 | 0.00046 | 0.000026 | 0.0649 | 0.400 | 163 | 47.9 |
| 15 | 0.0566 | 0.00275 | 0.000026 | 0.1409 | 0.402 | 708 | 292.9 |
| 16 | 0.0516 | 0.00207 | 0.000026 | 0.1345 | 0.384 | 606 | 262.5 |
| 17 | 0.0579 | 0.00288 | 0.000026 | 0.1339 | 0.432 | 412 | 172.4 |
| 20 | 0.0382 | -0.00037 | 0.000026 | 0.0848 | 0.451 | 77 | 41.1 |
| 21 | 0.0474 | 0.00146 | 0.000027 | 0.1213 | 0.391 | 540 | 235.9 |
| 24 | 0.0335 | 0.00113 | 0.000028 | 0.0944 | 0.355 | 275 | 98.5 |
| 25 | 0.0430 | 0.00177 | 0.000027 | 0.1083 | 0.397 | 549 | 213.8 |
| 27 | 0.0577 | 0.00291 | 0.000026 | 0.1351 | 0.428 | 211 | 83.4 |
| 29 | 0.0431 | 0.00176 | 0.000028 | 0.1107 | 0.390 | 56 | 27.6 |

