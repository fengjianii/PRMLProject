"""
Reproducible training script with best hyperparameters
Generated: 2026-06-03 12:57:01
Uses project MeowModel with update_params() to apply optimized params.
"""

from mdl import MeowModel
from eval import MeowEvaluator

BEST_PARAMS = {
  "max_depth": 3,
  "num_leaves": 8,
  "learning_rate": 0.04155397855708038,
  "n_estimators": 1500,
  "subsample": 0.7313811040057838,
  "colsample_bytree": 0.42221339552022713,
  "min_child_samples": 150,
  "reg_alpha": 0.018476435128841204,
  "reg_lambda": 0.9683377710407788
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
