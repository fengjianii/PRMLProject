import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from log import log


class MeowModel(object):
    """Prediction model for 12-minute forward return.

    Supports two model types:
      - 'ridge':   Ridge linear regression (baseline, fast, interpretable)
      - 'gbt':     Gradient Boosted Trees (non-linear, better performance)

    Default is 'gbt' which captures non-linear interactions between
    price, order-book, and trade features.
    """

    def __init__(self, cacheDir, model_type="gbt"):
        self.model_type = model_type
        if model_type == "ridge":
            self.estimator = Ridge(
                alpha=0.5,
                random_state=None,
                fit_intercept=False,
                tol=1e-8,
            )
        elif model_type == "gbt":
            self.estimator = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=300,
                max_depth=8,
                min_samples_leaf=50,
                max_leaf_nodes=63,
                random_state=42,
                early_stopping=False,
                validation_fraction=None,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, xdf, ydf):
        X = xdf.to_numpy().astype(np.float32)
        y = ydf.to_numpy().ravel().astype(np.float32)
        log.inf(f"Fitting {self.model_type} model on {X.shape[0]:,} samples "
                f"with {X.shape[1]} features...")
        self.estimator.fit(X, y)
        log.inf("Done fitting")

    def predict(self, xdf):
        X = xdf.to_numpy().astype(np.float32)
        return self.estimator.predict(X)
