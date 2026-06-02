"""
Reproducible training script with best hyperparameters
Generated: 2026-06-02 13:01:57
Uses project MeowModel with update_params() to apply optimized params.
"""

from mdl import MeowModel
from eval import MeowEvaluator

BEST_PARAMS = {
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
