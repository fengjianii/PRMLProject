# LightGBM Auto-Tuning Report
Generated: 2026-06-02 13:01:56

## 1. Tuning Summary
- Algorithm: bayesian
- Total Trials: 30
- Completed Trials: 20
- Total Time: 1574.7s (0.44h)
- Best Pearson: 0.0532

## 2. Best Hyperparameters
```json
{
  "max_depth": 5,
  "num_leaves": 14,
  "learning_rate": 0.032550900171720624,
  "n_estimators": 800,
  "subsample": 0.7069023275635302,
  "colsample_bytree": 0.7687201381729933,
  "min_child_samples": 100,
  "reg_alpha": 0.012121000362935315,
  "reg_lambda": 0.3503776504888305
}
```

## 3. Performance Improvement
- Baseline Pearson: 0.0576
- Optimized Pearson: 0.0532
- Improvement: -0.0044 (-7.6%)
- Target Pearson: 0.09
- Target Reached: No

## 4. Top 5 Models
| Rank | Pearson | R2 | MSE | Key Params |
|------|---------|-----|-----|------------|
| 1 | 0.0532 | 0.00234 | 0.000026 | depth=5, lr=0.032550900171720624, leaves=14 |
| 2 | 0.0452 | 0.00203 | 0.000027 | depth=4, lr=0.023746198818402044, leaves=10 |
| 3 | 0.0445 | 0.00061 | 0.000027 | depth=3, lr=0.04155397855708038, leaves=8 |
| 4 | 0.0418 | 0.00128 | 0.000027 | depth=4, lr=0.005746499650110705, leaves=12 |
| 5 | 0.0411 | -0.00002 | 0.000028 | depth=3, lr=0.047181627714860244, leaves=7 |

## 5. Parameter Importance
| Parameter | Importance |
|-----------|------------|
| reg_lambda | 0.4305 |
| subsample | 0.2814 |
| n_estimators | 0.0946 |
| colsample_bytree | 0.0847 |
| min_child_samples | 0.0528 |
| learning_rate | 0.0244 |
| reg_alpha | 0.0200 |
| max_depth | 0.0116 |

## 6. Overfitting Analysis
- Best trial overfitting ratio (val_pearson/train_pearson): 0.456
- Overfitting check (ratio >= 0.9): Warning
- Train Pearson: 0.1168
- Val Pearson: 0.0532

## 7. How to Apply Best Params
```python
from mdl import MeowModel
model = MeowModel(cacheDir=None)
model.load_params_from_json('tuning_output/best_params.json')
# or manually:
model.update_params({
    'max_depth': 5,
    'num_leaves': 14,
    'learning_rate': 0.032550900171720624,
    'n_estimators': 800,
    'subsample': 0.7069023275635302,
    'colsample_bytree': 0.7687201381729933,
    'min_child_samples': 100,
    'reg_alpha': 0.012121000362935315,
    'reg_lambda': 0.3503776504888305,
})
model.fit(xdf, ydf)
```

## 8. Trial Details
| Trial | Pearson | R2 | MSE | TrainPearson | OverfitRatio | BestIter | Time(s) |
|-------|---------|-----|-----|-------------|-------------|----------|--------|
| 1 | 0.0352 | 0.00047 | 0.000026 | 0.0988 | 0.357 | 70 | 27.0 |
| 2 | 0.0378 | 0.00126 | 0.000025 | 0.0962 | 0.393 | 155 | 51.0 |
| 3 | 0.0418 | 0.00128 | 0.000027 | 0.0767 | 0.545 | 562 | 180.5 |
| 4 | 0.0394 | 0.00095 | 0.000027 | 0.0601 | 0.656 | 290 | 87.1 |
| 5 | 0.0452 | 0.00203 | 0.000027 | 0.0898 | 0.503 | 413 | 124.0 |
| 6 | 0.0349 | 0.00086 | 0.000026 | 0.0995 | 0.351 | 238 | 64.1 |
| 8 | 0.0445 | 0.00061 | 0.000027 | 0.0906 | 0.491 | 205 | 55.6 |
| 11 | 0.0311 | -0.00001 | 0.000026 | 0.0718 | 0.433 | 338 | 96.1 |
| 12 | 0.0401 | 0.00016 | 0.000027 | 0.0871 | 0.461 | 144 | 41.0 |
| 13 | 0.0399 | 0.00132 | 0.000027 | 0.0959 | 0.416 | 375 | 72.1 |
| 15 | 0.0298 | 0.00114 | 0.000028 | 0.0706 | 0.422 | 212 | 50.7 |
| 17 | 0.0532 | 0.00234 | 0.000026 | 0.1168 | 0.456 | 166 | 51.9 |
| 18 | 0.0389 | 0.00092 | 0.000028 | 0.1100 | 0.354 | 104 | 38.3 |
| 19 | 0.0396 | 0.00057 | 0.000027 | 0.0755 | 0.524 | 258 | 101.7 |
| 20 | 0.0386 | 0.00148 | 0.000028 | 0.0802 | 0.481 | 333 | 96.5 |
| 21 | 0.0264 | 0.00093 | 0.000029 | 0.0701 | 0.377 | 57 | 30.8 |
| 22 | 0.0236 | 0.00081 | 0.000029 | 0.0716 | 0.330 | 29 | 19.0 |
| 23 | 0.0411 | -0.00002 | 0.000028 | 0.0861 | 0.477 | 180 | 49.4 |
| 25 | 0.0326 | 0.00070 | 0.000029 | 0.0643 | 0.508 | 104 | 41.9 |
| 29 | 0.0363 | 0.00080 | 0.000030 | 0.0662 | 0.548 | 142 | 50.6 |

