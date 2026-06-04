from meow import MeowEngine

engine = MeowEngine(h5dir="../../archive", cacheDir=None, feature_set='full')

best_params = engine.tune(
    20230601, 20231130,
    n_trials=60,
    algorithm='bayesian',
    n_splits=3,
    early_stopping_rounds=50,
    timeout_per_trial=1200,
    apply_best=True,
    sample_ratio=0.8,
)

engine.fit(20230601, 20231130)
engine.eval(20231201, 20231229)
