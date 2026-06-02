"""
Reproducible training script with best hyperparameters
Generated: 2026-06-02 15:35:06
Uses project MeowModel with update_params() to apply optimized params.
"""

from mdl import MeowModel
from eval import MeowEvaluator

BEST_PARAMS = {
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
