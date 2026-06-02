"""
Reproducible training script with best hyperparameters
Generated: 2026-06-02 17:10:58
Uses project MeowModel with update_params() to apply optimized params.
"""

from mdl import MeowModel
from eval import MeowEvaluator

BEST_PARAMS = {
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
