from meow import MeowEngine

engine = MeowEngine(h5dir="../../archive", cacheDir=None)

# 自动调参
best_params = engine.tune(
    20230601, 20231130,
    n_trials=30,
    algorithm='bayesian',
    n_splits=5,
    early_stopping_rounds=50,
    timeout_per_trial=600,
    apply_best=True,
    sample_ratio=0.5,
)

# 用最优参数训练
engine.fit(20230601, 20231130)

# 在测试集上评估
engine.eval(20231201, 20231229)