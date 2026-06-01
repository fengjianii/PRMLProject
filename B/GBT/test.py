import lightgbm as lgb
import numpy as np

print(f"LightGBM 版本: {lgb.__version__}")

# 生成测试数据
X = np.random.rand(1000, 20).astype(np.float32)
y = np.random.rand(1000).astype(np.float32)

params_gpu = {
    'device': 'gpu',        # 使用 OpenCL
    'gpu_device_id': 0,
    'max_bin': 63,
    'gpu_use_dp': False,
    'verbose': -1,
    'num_iterations': 10
}

try:
    model = lgb.LGBMRegressor(**params_gpu)
    model.fit(X, y)
    print("✅ OpenCL (GPU) 模式训练成功！")
except Exception as e:
    print(f"❌ 训练失败: {e}")