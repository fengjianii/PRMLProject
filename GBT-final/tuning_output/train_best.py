"""
Reproducible training script with best hyperparameters
Generated: 2026-06-04 01:21:18
Uses project MeowModel with update_params() to apply optimized params.
"""

from mdl import MeowModel
from eval import MeowEvaluator

BEST_PARAMS = {
  "max_depth": 5,
  "num_leaves": 26,
  "learning_rate": 0.03585096884900132,
  "n_estimators": 1000,
  "subsample": 0.5701269205911143,
  "colsample_bytree": 0.6033615848207509,
  "min_child_samples": 250,
  "reg_alpha": 1.0488862557637348,
  "reg_lambda": 0.2693466283225946
}

def train_best_model(xdf, ydf):
    model = MeowModel(cacheDir=None)
    model.update_params(BEST_PARAMS)
    model.update_params({'early_stopping_rounds': 50})
    model.fit(xdf, ydf)
    return model

if __name__ == '__main__':
    # Load your data using project modules
    # from dl import MeowDataLoader
    # from feat import MeowFeatureGenerator
    # ... load and generate features ...
    # model = train_best_model(xdf, ydf)
    print("Load data and call train_best_model(xdf, ydf)")
