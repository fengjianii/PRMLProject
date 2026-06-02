......................# 调参策略记录

## V1 (2026-06-02 初版)

### 搜索算法
- Optuna TPE贝叶斯优化，30轮，seed=42
- SQLite持久化

### 搜索空间（混合连续+离散）
| 参数 | 范围 | 类型 |
|------|------|------|
| max_depth | 3~6 | int |
| num_leaves | 7~2^max_depth | int |
| learning_rate | 0.005~0.05 | float(log) |
| n_estimators | 200~1000(step=100) | int |
| subsample | 0.5~0.8 | float |
| colsample_bytree | 0.5~0.8 | float |
| min_child_samples | 50~500(step=50) | int |
| reg_alpha | 0.01~5.0 | float(log) |
| reg_lambda | 0.01~5.0 | float(log) |

### 评估策略
- 5折 TimeSeriesSplit
- 每trial独立随机采样子集(sample_ratio=0.5)
- 提前剪枝阈值: OverfitRatio < 0.3
- 过拟合惩罚: OverfitRatio < 0.5 → Pearson×OverfitRatio
- 早停: 50轮

### 时间控制
- 单trial超时600秒

### 结果
- Best Pearson: 0.0577 (基线0.0576, +0.2%)
- 23/30轮完成, 7轮提前剪枝
- 最优参数: depth=5, leaves=7, lr=0.047, n_est=900, subsample=0.56, colsample=0.55, min_child=100, reg_alpha=0.066, reg_lambda=0.26

### 诊断问题
1. **5折CV第1折数据太少**: 第1折只用20%数据训练，Pearson=0.0频繁出现，污染均值
2. **lr搜索范围偏小**: 最优lr=0.047已触及0.05上限，0.05~0.1区间未探索
3. **max_depth=6未被利用**: 最优num_leaves=7说明depth=5~6的额外深度没被用
4. **OverfitRatio普遍0.3~0.6**: 验证性能不到训练一半，信号弱+过拟合，调参难突破
5. **早停50轮对小lr太激进**: lr=0.005时50轮内损失可能还在下降

---

## V2 (2026-06-02 优化版)

### 搜索算法
- Optuna TPE贝叶斯优化，30轮，seed=42
- SQLite持久化

### 搜索空间
| 参数 | 范围 | 类型 | 变更说明 |
|------|------|------|----------|
| max_depth | 3~5 | int | 去掉6，最优leaves=7说明深度6没被利用 |
| num_leaves | 7~2^max_depth | int | 不变 |
| learning_rate | 0.005~0.1 | float(log) | 扩展上限0.05→0.1，覆盖基线lr=0.1 |
| n_estimators | 200~1000(step=100) | int | 不变 |
| subsample | 0.5~0.8 | float | 不变 |
| colsample_bytree | 0.5~0.8 | float | 不变 |
| min_child_samples | 50~500(step=50) | int | 不变 |
| reg_alpha | 0.01~5.0 | float(log) | 不变 |
| reg_lambda | 0.01~5.0 | float(log) | 不变 |

### 评估策略
- **3折 TimeSeriesSplit** (5→3，第1折用50%数据而非20%)
- 每trial独立随机采样子集(sample_ratio=0.5)
- 提前剪枝阈值: OverfitRatio < 0.3
- 过拟合惩罚: OverfitRatio < 0.5 → Pearson×OverfitRatio
- **早停轮数与lr联动**: early_stopping = max(50, round(0.1/learning_rate * 50))
  - lr=0.1 → 50轮
  - lr=0.01 → 500轮
  - lr=0.005 → 1000轮

### 时间控制
- 单trial超时600秒

### 预期改进
1. 3折CV第1折训练数据从20%→50%，减少Pearson=0.0污染
2. lr扩展到0.1，覆盖基线参数区域
3. max_depth去掉6，减少搜索空间
4. 早停与lr联动，小lr不会被过早停止

### 结果
- Best Pearson: 0.0546 (基线0.0576, -5.3%)
- 28/30轮完成, 2轮提前剪枝
- 正式训练测试集Pearson: 0.0553 (基线0.0576, -4.0%)
- 最优参数: depth=4, leaves=9, lr=0.025, n_est=300, subsample=0.74, colsample=0.52, min_child=500, reg_alpha=1.21, reg_lambda=0.034

### 诊断问题
1. **正则化过重导致欠拟合**: reg_alpha=1.21压掉大量特征，min_child=500让树几乎长不出来
2. **过拟合惩罚机制有害**: 金融预测OverfitRatio 0.4~0.6是正常水平，惩罚它让TPE走向极端保守
3. **n_estimators=300太少**: 配合重正则化，模型容量严重不足
4. **Top5全部低于基线0.0576**: 搜索方向整体偏离

---

## V3 (2026-06-02 去惩罚+收窄正则化)

### 搜索算法
- Optuna TPE贝叶斯优化，30轮，seed=42
- SQLite持久化

### 搜索空间
| 参数 | 范围 | 类型 | 变更说明 |
|------|------|------|----------|
| max_depth | 3~5 | int | 不变 |
| num_leaves | 7~2^max_depth | int | 不变 |
| learning_rate | 0.005~0.1 | float(log) | 不变 |
| n_estimators | 200~1000(step=100) | int | 不变 |
| subsample | 0.5~0.8 | float | 不变 |
| colsample_bytree | 0.5~0.8 | float | 不变 |
| min_child_samples | 50~300(step=50) | int | 上限500→300，V2最优500太大导致欠拟合 |
| reg_alpha | 0.01~2.0 | float(log) | 上限5→2，V2最优1.21太重 |
| reg_lambda | 0.01~2.0 | float(log) | 上限5→2，与reg_alpha对称收窄 |

### 评估策略
- 3折 TimeSeriesSplit
- 每trial独立随机采样子集(sample_ratio=0.5)
- 提前剪枝阈值: OverfitRatio < 0.3（保留，过滤极端过拟合）
- **去掉过拟合惩罚**: 不再对OverfitRatio<0.5乘以惩罚系数
  - V2证明惩罚让TPE走向极端保守(重正则化+少树)，导致欠拟合
  - 金融预测OverfitRatio 0.4~0.6是正常水平，不应惩罚
  - 提前剪枝(ratio<0.3)已足够过滤极端过拟合
- 早停轮数与lr联动: early_stopping = max(50, round(0.1/learning_rate * 50))

### 时间控制
- 单trial超时600秒

### 预期改进
1. 去掉过拟合惩罚，TPE纯粹优化Pearson，不人为压低
2. reg_alpha/min_child上限收窄，避免TPE走向极端保守
3. 搜索空间更集中在合理区间