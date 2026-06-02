# LightGBM Auto-Tuning Report
Generated: 2026-06-02 15:35:06

## 1. Tuning Summary
- Algorithm: bayesian
- Total Trials: 30
- Completed Trials: 23
- Total Time: 2514.3s (0.70h)
- Best Pearson: 0.0541

## 2. Best Hyperparameters
```json
{
  "max_depth": 4,
  "num_leaves": 10,
  "learning_rate": 0.007712667480772209,
  "n_estimators": 600,
  "subsample": 0.6913954997948364,
  "colsample_bytree": 0.6804887419171513,
  "min_child_samples": 200,
  "reg_alpha": 0.3998597050466127,
  "reg_lambda": 0.05145174567726845
}
```

## 3. Performance Improvement
- Baseline Pearson: 0.0576
- Optimized Pearson: 0.0541
- Improvement: -0.0035 (-6.1%)
- Target Pearson: 0.09
- Target Reached: No

## 4. Top 5 Models
| Rank | Pearson | R2 | MSE | Key Params |
|------|---------|-----|-----|------------|
| 1 | 0.0541 | 0.00228 | 0.000029 | depth=4, lr=0.007712667480772209, leaves=10 |
| 2 | 0.0504 | 0.00210 | 0.000026 | depth=4, lr=0.025411709798607296, leaves=9 |
| 3 | 0.0487 | 0.00178 | 0.000026 | depth=4, lr=0.007602560195556776, leaves=10 |
| 4 | 0.0466 | 0.00178 | 0.000026 | depth=4, lr=0.005315374227132284, leaves=11 |
| 5 | 0.0462 | 0.00071 | 0.000027 | depth=5, lr=0.009671121579347167, leaves=32 |

## 5. Parameter Importance
| Parameter | Importance |
|-----------|------------|
| subsample | 0.3516 |
| reg_alpha | 0.3286 |
| max_depth | 0.0795 |
| n_estimators | 0.0663 |
| min_child_samples | 0.0608 |
| learning_rate | 0.0484 |
| colsample_bytree | 0.0428 |
| reg_lambda | 0.0220 |

## 6. Overfitting Analysis
- Best trial overfitting ratio (val_pearson/train_pearson): 0.589
- Overfitting check (ratio >= 0.9): Warning
- Train Pearson: 0.0918
- Val Pearson: 0.0541

## 7. How to Apply Best Params
```python
from mdl import MeowModel
model = MeowModel(cacheDir=None)
model.load_params_from_json('tuning_output/best_params.json')
# or manually:
model.update_params({
    'max_depth': 4,
    'num_leaves': 10,
    'learning_rate': 0.007712667480772209,
    'n_estimators': 600,
    'subsample': 0.6913954997948364,
    'colsample_bytree': 0.6804887419171513,
    'min_child_samples': 200,
    'reg_alpha': 0.3998597050466127,
    'reg_lambda': 0.05145174567726845,
})
model.fit(xdf, ydf)
```

## 8. Trial Details
| Trial | Pearson | R2 | MSE | TrainPearson | OverfitRatio | BestIter | Time(s) |
|-------|---------|-----|-----|-------------|-------------|----------|--------|
| 0 | 0.0410 | -0.00017 | 0.000028 | 0.0870 | 0.471 | 112 | 68.4 |
| 1 | 0.0430 | 0.00171 | 0.000027 | 0.0998 | 0.431 | 140 | 53.1 |
| 2 | 0.0413 | 0.00063 | 0.000027 | 0.0979 | 0.421 | 125 | 46.2 |
| 3 | 0.0298 | -0.00027 | 0.000031 | 0.0514 | 0.579 | 469 | 217.4 |
| 4 | 0.0359 | 0.00050 | 0.000025 | 0.0957 | 0.375 | 197 | 66.9 |
| 5 | 0.0342 | 0.00078 | 0.000026 | 0.0889 | 0.385 | 100 | 49.5 |
| 6 | 0.0371 | 0.00122 | 0.000028 | 0.0958 | 0.387 | 245 | 74.4 |
| 7 | 0.0504 | 0.00210 | 0.000026 | 0.0933 | 0.540 | 239 | 81.0 |
| 8 | 0.0377 | 0.00009 | 0.000026 | 0.0958 | 0.393 | 105 | 45.4 |
| 9 | 0.0218 | -0.00037 | 0.000026 | 0.0526 | 0.414 | 218 | 97.6 |
| 10 | 0.0430 | 0.00110 | 0.000027 | 0.1120 | 0.384 | 185 | 73.0 |
| 11 | 0.0354 | 0.00155 | 0.000027 | 0.0640 | 0.553 | 133 | 73.6 |
| 12 | 0.0412 | 0.00101 | 0.000028 | 0.1179 | 0.350 | 161 | 66.8 |
| 14 | 0.0309 | -0.00044 | 0.000027 | 0.0850 | 0.363 | 102 | 41.5 |
| 15 | 0.0462 | 0.00071 | 0.000027 | 0.1270 | 0.364 | 336 | 126.1 |
| 17 | 0.0246 | -0.00088 | 0.000026 | 0.0620 | 0.396 | 31 | 25.2 |
| 22 | 0.0422 | 0.00141 | 0.000027 | 0.1014 | 0.416 | 147 | 84.6 |
| 24 | 0.0443 | 0.00083 | 0.000027 | 0.0919 | 0.482 | 288 | 89.6 |
| 25 | 0.0341 | 0.00125 | 0.000027 | 0.0814 | 0.420 | 293 | 82.8 |
| 26 | 0.0541 | 0.00228 | 0.000029 | 0.0918 | 0.589 | 595 | 173.2 |
| 27 | 0.0487 | 0.00178 | 0.000026 | 0.0985 | 0.495 | 598 | 183.1 |
| 28 | 0.0361 | 0.00058 | 0.000027 | 0.0558 | 0.648 | 401 | 181.4 |
| 29 | 0.0466 | 0.00178 | 0.000026 | 0.0915 | 0.510 | 621 | 216.3 |

