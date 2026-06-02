# LightGBM Auto-Tuning Report
Generated: 2026-06-02 09:26:47

## 1. Tuning Summary
- Algorithm: bayesian
- Total Trials: 30
- Completed Trials: 23
- Total Time: 3430.6s (0.95h)
- Best Pearson: 0.0577

## 2. Best Hyperparameters
```json
{
  "max_depth": 5,
  "num_leaves": 7,
  "learning_rate": 0.04665303012212833,
  "n_estimators": 900,
  "subsample": 0.5637017332034828,
  "colsample_bytree": 0.5545474901621302,
  "min_child_samples": 100,
  "reg_alpha": 0.06624310605949987,
  "reg_lambda": 0.2607965659809584
}
```

## 3. Performance Improvement
- Baseline Pearson: 0.0576
- Optimized Pearson: 0.0577
- Improvement: +0.0001 (+0.2%)
- Target Pearson: 0.09
- Target Reached: No

## 4. Top 5 Models
| Rank | Pearson | R2 | MSE | Key Params |
|------|---------|-----|-----|------------|
| 1 | 0.0577 | 0.00239 | 0.000026 | depth=5, lr=0.04665303012212833, leaves=7 |
| 2 | 0.0528 | 0.00196 | 0.000026 | depth=3, lr=0.025458179729092773, leaves=8 |
| 3 | 0.0502 | 0.00123 | 0.000026 | depth=4, lr=0.024165903162442326, leaves=7 |
| 4 | 0.0484 | 0.00177 | 0.000032 | depth=4, lr=0.03121374753761447, leaves=10 |
| 5 | 0.0454 | 0.00155 | 0.000026 | depth=4, lr=0.026975154833351143, leaves=16 |

## 5. Parameter Importance
| Parameter | Importance |
|-----------|------------|
| learning_rate | 0.2277 |
| subsample | 0.2103 |
| min_child_samples | 0.1602 |
| max_depth | 0.1157 |
| n_estimators | 0.0957 |
| colsample_bytree | 0.0824 |
| reg_lambda | 0.0699 |
| reg_alpha | 0.0380 |

## 6. Overfitting Analysis
- Best trial overfitting ratio (val_pearson/train_pearson): 0.518
- Overfitting check (ratio >= 0.9): Warning
- Train Pearson: 0.1114
- Val Pearson: 0.0577

## 7. How to Apply Best Params
```python
from mdl import MeowModel
model = MeowModel(cacheDir=None)
model.load_params_from_json('tuning_output/best_params.json')
# or manually:
model.update_params({
    'max_depth': 5,
    'num_leaves': 7,
    'learning_rate': 0.04665303012212833,
    'n_estimators': 900,
    'subsample': 0.5637017332034828,
    'colsample_bytree': 0.5545474901621302,
    'min_child_samples': 100,
    'reg_alpha': 0.06624310605949987,
    'reg_lambda': 0.2607965659809584,
})
model.fit(xdf, ydf)
```

## 8. Trial Details
| Trial | Pearson | R2 | MSE | TrainPearson | OverfitRatio | BestIter | Time(s) |
|-------|---------|-----|-----|-------------|-------------|----------|--------|
| 0 | 0.0454 | 0.00155 | 0.000026 | 0.0791 | 0.574 | 362 | 191.3 |
| 1 | 0.0577 | 0.00239 | 0.000026 | 0.1114 | 0.518 | 287 | 120.2 |
| 2 | 0.0335 | 0.00122 | 0.000027 | 0.0496 | 0.675 | 183 | 102.8 |
| 3 | 0.0345 | -0.00097 | 0.000028 | 0.0889 | 0.388 | 414 | 251.3 |
| 4 | 0.0502 | 0.00123 | 0.000026 | 0.0854 | 0.588 | 363 | 147.9 |
| 5 | 0.0260 | -0.00012 | 0.000026 | 0.0666 | 0.391 | 131 | 103.5 |
| 7 | 0.0272 | -0.00078 | 0.000028 | 0.0578 | 0.471 | 136 | 70.5 |
| 8 | 0.0528 | 0.00196 | 0.000026 | 0.0988 | 0.534 | 410 | 161.4 |
| 9 | 0.0361 | 0.00050 | 0.000028 | 0.0568 | 0.636 | 304 | 190.3 |
| 10 | 0.0363 | 0.00080 | 0.000027 | 0.1162 | 0.313 | 54 | 54.8 |
| 11 | 0.0427 | 0.00124 | 0.000029 | 0.0723 | 0.591 | 238 | 118.2 |
| 12 | 0.0279 | -0.00035 | 0.000026 | 0.0646 | 0.432 | 177 | 101.0 |
| 14 | 0.0424 | 0.00184 | 0.000027 | 0.0696 | 0.610 | 320 | 185.1 |
| 15 | 0.0420 | 0.00048 | 0.000028 | 0.1270 | 0.331 | 244 | 200.0 |
| 16 | 0.0390 | 0.00119 | 0.000025 | 0.0936 | 0.417 | 223 | 135.8 |
| 17 | 0.0412 | 0.00060 | 0.000033 | 0.0722 | 0.571 | 430 | 230.5 |
| 20 | 0.0434 | 0.00081 | 0.000033 | 0.0752 | 0.578 | 233 | 127.0 |
| 22 | 0.0436 | 0.00138 | 0.000026 | 0.0807 | 0.541 | 287 | 140.1 |
| 23 | 0.0272 | -0.00152 | 0.000028 | 0.0589 | 0.462 | 79 | 53.8 |
| 24 | 0.0415 | 0.00100 | 0.000026 | 0.0675 | 0.614 | 297 | 132.6 |
| 27 | 0.0305 | -0.00043 | 0.000028 | 0.0708 | 0.431 | 260 | 164.3 |
| 28 | 0.0368 | 0.00052 | 0.000026 | 0.0586 | 0.628 | 288 | 133.8 |
| 29 | 0.0484 | 0.00177 | 0.000032 | 0.0839 | 0.577 | 306 | 168.2 |

