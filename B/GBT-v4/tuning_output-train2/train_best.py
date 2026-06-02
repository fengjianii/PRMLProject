"""
Reproducible training script with best hyperparameters
Generated: 2026-06-02 10:36:27
Uses project MeowModel with update_params() to apply optimized params.
"""

from mdl import MeowModel
from eval import MeowEvaluator

BEST_PARAMS = {
  "max_depth": 4,
  "num_leaves": 9,
  "learning_rate": 0.025411709798607296,
  "n_estimators": 300,
  "subsample": 0.7406590942262119,
  "colsample_bytree": 0.5223651931039313,
  "min_child_samples": 500,
  "reg_alpha": 1.2141307774357368,
  "reg_lambda": 0.03438172512115175
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
