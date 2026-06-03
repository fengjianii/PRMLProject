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

### 结果
- Best Pearson: 0.0532 (CV评估, 基线0.0576, -7.6%)
- 正式训练测试集Pearson: 0.0606 (基线0.0576, +5.2%)
- 20/30轮完成, 10轮提前剪枝
- 最优参数: depth=5, leaves=14, lr=0.033, n_est=800, subsample=0.71, colsample=0.77, min_child=100, reg_alpha=0.012, reg_lambda=0.35

### 诊断
- CV评估(0.0532)与正式训练(0.0606)差距14%，说明sample_ratio=0.5子集调参低估模型性能
- 去惩罚后TPE不再走向极端保守，但CV评估不可靠导致搜索方向仍有偏差
- 三版调参CV均未超过基线，但正式训练均达到0.06左右，说明GBT模型在当前特征集上约0.06是天花板

---

## P1 特征工程：横截面rank标准化 (2026-06-02)

### 改动说明
对20个与股票规模/流动性相关的特征，在每个时间点(interval)做横截面rank标准化：
- `cs_rank_X = groupby("interval")[X].rank(pct=True)`
- 结果为0~1的排名百分位，消除股票间异质性

### 新增特征（20个）
cs_rank_ob_imb0, cs_rank_ob_imb4, cs_rank_ob_imb9,
cs_rank_trade_imb, cs_rank_trade_imbema5,
cs_rank_spread, cs_rank_relative_spread, cs_rank_weighted_spread_4,
cs_rank_amount_imb_4,
cs_rank_depth_sum_4, cs_rank_depth_delta_4,
cs_rank_buy_pressure_0, cs_rank_buy_pressure_4,
cs_rank_micro_price_dev, cs_rank_bid_ask_bias_0,
cs_rank_trade_buy_intensity, cs_rank_trade_net_intensity,
cs_rank_trade_buy_turnover_ratio,
cs_rank_trade_count_imb, cs_rank_buy_trade_size

### 交互特征改动
5个交互特征改用rank版本：
- ret_3_x_imb0: ret_3 × cs_rank_ob_imb0
- ret_6_x_imb0: ret_6 × cs_rank_ob_imb0
- ret_3_x_buy_intensity: ret_3 × cs_rank_trade_buy_intensity
- spread_x_vol: cs_rank_relative_spread × rolling_vol_12
- highlow_x_imb0: high_low_range × cs_rank_ob_imb0

### 总特征数
54 → 74

### 预期效果
- 消除股票间异质性，模型学到"相对强弱"而非"绝对值"
- Pearson预期提升 +0.01~0.03
- 效率影响：特征生成+5~10%，训练+10~15%，内存+37%

### 结果
- 调参Best Pearson: 0.0541 (CV评估, 基线0.0576, -6.1%)
- 正式训练测试集Pearson: 0.0546 (基线0.0576, -5.2%)
- **比V3(0.0606)更差**

### 诊断
1. **原始特征与rank特征冗余**: 74特征中20对高度相关，colsample被稀释
2. **rank特征在时序CV中行为不同**: 不同时间点rank=0.8对应不同股票
3. **lr=0.0077太小**: 冗余特征增加噪声，模型不敢大胆分裂

---

## P1-fix 特征工程：rank替换原始值 (2026-06-02)

### 改动说明
P1的问题：新增20个rank特征导致冗余。修复：用rank特征**替换**20个原始特征，保持54个特征数不变。

### 替换映射（20个）
| 原始特征 | 替换为 |
|----------|--------|
| ob_imb0 | cs_rank_ob_imb0 |
| ob_imb4 | cs_rank_ob_imb4 |
| ob_imb9 | cs_rank_ob_imb9 |
| trade_imb | cs_rank_trade_imb |
| trade_imbema5 | cs_rank_trade_imbema5 |
| spread | cs_rank_spread |
| relative_spread | cs_rank_relative_spread |
| weighted_spread_4 | cs_rank_weighted_spread_4 |
| amount_imb_4 | cs_rank_amount_imb_4 |
| depth_sum_4 | cs_rank_depth_sum_4 |
| depth_delta_4 | cs_rank_depth_delta_4 |
| buy_pressure_0 | cs_rank_buy_pressure_0 |
| buy_pressure_4 | cs_rank_buy_pressure_4 |
| micro_price_dev | cs_rank_micro_price_dev |
| bid_ask_bias_0 | cs_rank_bid_ask_bias_0 |
| trade_buy_intensity | cs_rank_trade_buy_intensity |
| trade_net_intensity | cs_rank_trade_net_intensity |
| trade_buy_turnover_ratio | cs_rank_trade_buy_turnover_ratio |
| trade_count_imb | cs_rank_trade_count_imb |
| buy_trade_size | cs_rank_buy_trade_size |

### 总特征数
保持54个不变

### 预期效果
- 消除冗余，colsample效率恢复
- 模型学到"相对强弱"，泛化更好
- 无额外内存/训练开销

### 结果
- 调参Best Pearson: 0.0579 (CV评估, 首次超过基线0.0576)
- 正式训练测试集Pearson: 0.0615 (比V3的0.0606提升1.5%)
- rank替换策略有效，Top3 Pearson: 0.0579, 0.0577, 0.0566
- 最优参数: depth=5, leaves=27, lr=0.018, n_est=900

---

## V4 最终调参 (2026-06-02)

### 搜索空间变更
| 参数 | V3 | V4 | 理由 |
|------|----|----|------|
| num_leaves | 7~2^max_depth | 7~min(2^max_depth, 63) | 最优27接近32上限 |
| n_estimators | 200~1000 | 200~2000 | 更多轮让大模型充分学习 |

### 调参配置变更
| 参数 | V3 | V4 | 理由 |
|------|----|----|------|
| n_trials | 30 | 50 | 最后一轮，多搜 |
| sample_ratio | 0.5 | 0.7 | 减少CV与正式训练差距 |
| timeout_per_trial | 600s | 900s | 更多数据+更多树 |

### 其余不变
- max_depth 3~5, lr 0.005~0.1, subsample 0.5~0.8, colsample 0.5~0.8
- min_child_samples 50~300, reg_alpha/lambda 0.01~2.0
- 3折CV, 无过拟合惩罚, 提前剪枝ratio<0.3
- 早停与lr联动

### 结果
- 调参Best Pearson: 0.0583 (CV评估)
- 正式训练测试集Pearson: 0.0635
- 最优参数: depth=5, leaves=31, lr=0.0156, n_est=1600, subsample=0.502, colsample=0.551, min_child=200, reg_alpha=0.779, reg_lambda=1.908

---

## 特征工程V2 (2026-06-03)

### 三项改动

#### 1. 修复shift/rolling跨天bug
- **问题**: shift()/rolling()/diff()按groupby("symbol")分组，当天最后一分钟与下一天第一分钟连在一起
- **修复**: 新增_per_session()方法，所有时序操作按(symbol, date)分组
- **影响**: bret12, ret_1/3/6/12/24, rolling_mean/vol, depth_delta, rolling_trade_qty, trade_imbema5, trade_count_imbema5, overnight_gap等全部修正

#### 2. 修正cs_rank分组粒度
- **问题**: groupby("interval")会把不同日期同一时刻(6月1日9:31和6月2日9:31)混在一起排名
- **修复**: 改为groupby(["date", "interval"])，保证只比较同一天同一时刻的所有股票
- **影响**: 20个cs_rank特征和8个P1横截面特征

#### 3. 新增22个特征（P0+P1）

**P0 滚动统计特征（14个）**:
| 特征 | 说明 |
|------|------|
| ob_imb0_roll_mean_12 | ob_imb0的12窗口均值 |
| ob_imb0_roll_std_12 | ob_imb0的12窗口标准差 |
| ob_imb0_roll_skew_12 | ob_imb0的12窗口偏度近似 |
| ob_imb4_roll_mean_12 | ob_imb4的12窗口均值 |
| ob_imb4_roll_std_12 | ob_imb4的12窗口标准差 |
| trade_imb_roll_mean_12 | trade_imb的12窗口均值 |
| trade_imb_roll_std_12 | trade_imb的12窗口标准差 |
| trade_imb_roll_skew_12 | trade_imb的12窗口偏度近似 |
| trade_imbema5_roll_mean_12 | trade_imbema5的12窗口均值 |
| trade_imbema5_roll_std_12 | trade_imbema5的12窗口标准差 |
| ob_imb0_change_6 | ob_imb0的6窗口变化率 |
| ob_imb4_change_6 | ob_imb4的6窗口变化率 |
| trade_imb_change_6 | trade_imb的6窗口变化率 |
| trade_imbema5_change_6 | trade_imbema5的6窗口变化率 |

**P1 横截面特征（8个）**:
| 特征 | 说明 |
|------|------|
| cs_rank_ret_3 | ret_3的横截面rank |
| cs_rank_ret_6 | ret_6的横截面rank |
| cs_rank_rolling_vol_12 | rolling_vol_12的横截面rank |
| cs_rank_rolling_mean_ret_12 | rolling_mean_ret_12的横截面rank |
| cs_rank_ret_3_change | cs_rank_ret_3的1步变化 |
| cs_rank_ret_6_change | cs_rank_ret_6的1步变化 |
| cs_rank_rolling_vol_12_change | cs_rank_rolling_vol_12的1步变化 |
| cs_rank_rolling_mean_ret_12_change | cs_rank_rolling_mean_ret_12的1步变化 |

### 总特征数
54 → 76

---

## V5 调参策略更新 (2026-06-03)

### 变更
| 参数 | V4 | V5 | 理由 |
|------|----|----|------|
| colsample_bytree | 0.5~0.8 | **0.4~0.7** | 76特征下0.8=选61个太多，收窄避免噪声特征 |

### 其余不变
- max_depth 3~5, num_leaves 7~min(2^max_depth,63), lr 0.005~0.1
- n_estimators 200~2000, subsample 0.5~0.8
- min_child_samples 50~300, reg_alpha/lambda 0.01~2.0
- 3折CV, 无过拟合惩罚, 提前剪枝ratio<0.3
- 早停与lr联动, n_trials=50, sample_ratio=0.7, timeout=900s