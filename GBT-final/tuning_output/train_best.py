"""
Reproducible training script with best hyperparameters
Generated: 2026-06-02 20:53:19
Uses project MeowModel with update_params() to apply optimized params.
"""

from mdl import MeowModel
from eval import MeowEvaluator

BEST_PARAMS = {
  "max_depth": 5,
  "num_leaves": 31,
  "learning_rate": 0.015599098069279144,
  "n_estimators": 1600,
  "subsample": 0.501860762937803,
  "colsample_bytree": 0.5511927937508286,
  "min_child_samples": 200,
  "reg_alpha": 0.7794356250236579,
  "reg_lambda": 1.9078647856602788
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
