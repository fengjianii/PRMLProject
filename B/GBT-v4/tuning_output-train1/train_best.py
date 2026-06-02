"""
Reproducible training script with best hyperparameters
Generated: 2026-06-02 09:26:48
Uses project MeowModel with update_params() to apply optimized params.
"""

from mdl import MeowModel
from eval import MeowEvaluator

BEST_PARAMS = {
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
