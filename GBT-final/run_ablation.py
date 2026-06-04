"""
消融实验：定位V5特征集退化的原因

V4基线：54特征 → 测试集0.0635
V5全量：76特征 → 测试集0.0626（用V4参数）

消融组合：
1. v4:       54特征（V4基线，无P0+P1）
2. no_p1:    68特征（V4+P0，不含P1横截面8个）
3. no_p0:    62特征（V4+P1，不含P0滚动统计14个）
4. full:     76特征（V5全量）

每组用V4参数结构训练，在测试集评估，对比Pearson
"""

from meow import MeowEngine
from mdl import MeowModel
import json

V4_PARAMS = {
    'max_depth': 5,
    'num_leaves': 31,
    'learning_rate': 0.015599098069279144,
    'n_estimators': 1600,
    'subsample': 0.501860762937803,
    'colsample_bytree': 0.5511927937508286,
    'min_child_samples': 200,
    'min_child_weight': 1e-3,
    'reg_alpha': 0.7794356250236579,
    'reg_lambda': 1.9078647856602788,
    'subsample_freq': 1,
}

results = {}

for feature_set in ['v4', 'no_p1', 'no_p0', 'full']:
    print(f"\n{'='*60}")
    print(f"Feature set: {feature_set}")
    print(f"{'='*60}")
    
    engine = MeowEngine(h5dir="../../archive", cacheDir=None, feature_set=feature_set)
    engine.model.update_params(V4_PARAMS)
    engine.model.early_stopping_rounds = 100
    
    engine.fit(20230601, 20231130)
    engine.eval(20231201, 20231229)
    
    results[feature_set] = "see eval output above"

print(f"\n{'='*60}")
print("Ablation complete. Compare Pearson across feature sets.")
print(f"{'='*60}")