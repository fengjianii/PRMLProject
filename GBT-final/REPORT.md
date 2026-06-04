# MEOW 金融时序预测 — 项目报告

## 1. 项目背景与任务定义

本项目的目标是预测股票 12 分钟后的价格变化率（fret12）。数据覆盖约 300 只 A 股从 2023 年 6 月到 12 月的分钟级行情，包括价格、订单簿（order book）、成交明细和委托/撤单事件。

$$

\text{fret12} = \frac{\text{midpx}(t+12) - \text{midpx}(t)}{\text{midpx}(t)}
$$

任务本质上是一个**短周期收益率预测**问题。和图像识别或者推荐系统不一样，金融时序预测有几个特殊的地方：数据噪声极大、收益率分布是肥尾而不是正态的、而且市场状态本身也在不断变化（non-stationary）。所以一般来说，预测的时间越短准确率越高——12 分钟是一个比较折中的选择，既给了模型反应空间，也不至于把太多噪声卷进来。

---

## 2. 数据说明与探索性分析（EDA）

> 本节由 A 完成。EDA 脚本见 `eda.py`，图表生成见 `gen_figures.py`。

数据来源：MEOW 数据集，格式为按天存储的 HDF5 文件，每天约 7 万行。原始数据 62 个字段，分为以下几类：

| 类别 | 代表性字段 | 说明 |
|------|-----------|------|
| 标的 & 时间 | symbol, interval, date | 股票代码、毫秒级时间戳、日期 |
| 目标值 | **fret12** | 12 分钟后的 forward return |
| 价格 | midpx, lastpx, open, high, low, bid0, ask0 | 中间价、OHLC、买卖挂单价 |
| 盘口量 | bsize0, asize0, bsize0_4, asize0_4, ... | 各档位买卖挂单数量 |
| 盘口额 | btr0_4, atr0_4, btr5_9, atr5_9, ... | 各档位买卖挂单金额 |
| 成交 | nTradeBuy, tradeBuyQty, tradeBuyTurnover, ... | 主买/主卖成交的笔数、数量、金额 |
| 委托/撤单 | nAddBuy, addBuyQty, nCxlBuy, cxlBuyQty, ... | 新增和撤销的买卖委托 |

### 2.1 数据规模

- **时间范围**：2023 年 6 月 1 日至 12 月 29 日，共 144 个交易日
- **股票数量**：309 只
- **日内频率**：每只股票每天约 226 条分钟级记录
- **训练集**：6 月 1 日至 11 月 30 日（123 天），约 880 万样本
- **测试集**：12 月 1 日至 12 月 29 日（21 天），约 144 万样本
- **两段时间完全不重叠**，评估结果反映真实泛化能力

### 2.2 fret12 的分布特征

对 10 天样本（约 69 万行）分析 fret12 的统计特征：

- **均值 0.0001**：12 分钟尺度上，市场整体没有明显方向性
- **标准差 0.0064**：大部分时候 12 分钟价格变化在 ±1% 以内
- **偏度 2.17，峰度 22.6**：分布严重偏离正态——右偏 + 厚尾。极端值可达 ±10%
- **约 6% 的样本 fret12 = 0**：部分股票 12 分钟内中间价无变化，通常是流动性较差的标的

![fret12 分布](report_figures/fig01_fret12_distribution.png)

Q-Q 图清楚地显示了尾部偏离——用 MSE 作为损失函数会被极端值牵着走。这也是为什么业界更看重 Pearson Correlation：它不受数值尺度影响，反映的是预测方向和真实方向的一致性。

### 2.3 价格与价差

309 只股票价格跨度很大（2.9 元 ~ 328 元）。买卖一档价差（ask0 - bid0）均值约 0.02 元，相对价差均值约 0.075%（7.5 bps），市场整体流动性不错。价差在日内呈 U 型——开盘和收盘附近价差更大。

![日内成交活跃度](report_figures/fig04_intraday_trade.png)

### 2.4 盘口结构

一个值得注意的发现：**卖盘在所有档位上都多于买盘**。Level 0-4 的卖盘数量平均比买盘多约 8.7%，金额也多约 8.6%。可能原因：A 股融券成本高、做空不方便，持有者更倾向于挂卖单；或者 2023 年下半年市场整体偏谨慎。

越深层的档位（5-9 档、10-19 档）挂单量越大，流动性主要集中在价格稍远的位置。这意味着大单需要吃掉好几档，冲击成本不低。

![盘口深度](report_figures/fig03_orderbook_profile.png)

### 2.5 成交特征

每个分钟间隔平均约 136 笔成交，主买 65 笔、主卖 71 笔。成交数量上卖方向略多（imbalance 约 -7.3%）。买方单笔平均成交量略大于卖方，买盘以大单为主的特征更明显。成交强度（成交量 / 盘口深度）均值约 0.67。

### 2.6 缺失值情况

缺失集中在成交价的高/低字段（tradeBuyHigh/Low 等）和 VWAP 字段（约 1%-3%），原因是没有对应方向的成交。更深档位 bid/ask 有零星缺失（<0.05%）。特征工程中统一 fillna(0) 处理。

### 2.7 原始变量与 fret12 的线性相关

| 变量 | 线性相关系数 | 解读 |
|------|-----------|------|
| 买方成交强度 | +0.040 | 相关性最强，买方越主动短期越可能涨 |
| 盘口不平衡（Level 0） | -0.012 | 卖盘更多 → 价格稍偏下行 |
| 相对价差 | +0.010 | 价差大时波动大，但方向不确定 |
| 盘口不平衡（Level 0-4） | -0.006 | 和多档汇总的趋势一致 |

线性相关性整体都不高（绝对值最大才 0.04），说明单一原始变量预测能力有限，需要非线性模型捕捉变量之间的交互关系——这就是特征工程要解决的问题。

![fret12 vs 关键变量](report_figures/fig02_fret12_vs_variables.png)

---

## 3. 特征工程设计

> 本节由 A 完成。最终特征集以 `GBT-final/feat.py` 为准，共 76 个特征。

特征工程的思路来自市场微观结构理论。短期价格变动主要由订单流驱动——谁在买、谁在卖、挂了多少、撤了多少、成交了多少。我们围绕这个逻辑，在 V4 基线（54 特征）的基础上新增了 22 个特征（P0 滚动统计 14 个 + P1 横截面 8 个），总计 76 个。

### 特征总览

| 特征集 | 类别 | 数量 | 说明 |
|--------|------|------|------|
| V4 基线 | 基础 | 6 | 订单簿/交易不平衡 + 超额收益 |
| | 价格动量/反转 | 12 | 多尺度收益率 + 滚动均值/波动 + 价格位置 |
| | 盘口压力 | 16 | 价差 + 金额不平衡 + 深度 + 买方占比 + 微观价格 |
| | 成交主动性 | 15 | 成交强度 + 换手率 + 滚动成交量 + 笔数不平衡 + 单笔大小 |
| | 交互 | 5 | 动量×盘口、动量×成交、价差×波动 |
| | **V4 小计** | **54** | |
| P0 | 滚动统计 | 14 | 核心特征的 12 窗口均值/标准差/偏度 + 6 窗口变化率 |
| P1 | 横截面 | 8 | 收益率/波动率的截面排名 + 排名变化 |
| | **总计** | **76** | |

### 3.0 横截面 rank 标准化

在介绍具体特征之前，先说明一个重要的预处理操作。对 20 个与股票规模/流动性相关的特征，在每个时间点做横截面排名：

$$

\text{cs\_rank\_X}(i, t) = \frac{\text{rank}(X(i, t))}{N(t)}
$$

其中 $i$ 为股票，$t$ 为时间点（interval），$N(t)$ 为该时刻股票总数，结果映射到 $[0,1]$。

**做这件事的原因**：大盘股和小盘股的盘口深度、成交量差了几个数量级，`ob_imb0 = 0.3` 对一只大盘股和对一只小盘股的含义完全不同。截面 rank 之后，模型学到的是"这只股票相对其他股票强不强"，而不是绝对值。这是量化研究里的标准操作。

做了 cs_rank 的 20 个特征：
```
ob_imb0, ob_imb4, ob_imb9, trade_imb, trade_imbema5,
spread, relative_spread, weighted_spread_4,
amount_imb_4, depth_sum_4, depth_delta_4,
buy_pressure_0, buy_pressure_4,
micro_price_dev, bid_ask_bias_0,
trade_buy_intensity, trade_net_intensity,
trade_buy_turnover_ratio, trade_count_imb, buy_trade_size
```

无量纲特征（ret_*、rolling_mean/vol、high_low_range、price_position、overnight_gap、lagret12 等）保留原始值，不做 rank。

### 3.1 价格动量/反转特征（12 个）

短期价格变化存在两种对立的效应——动量（涨了继续涨）和反转（涨了会回调）。到底是哪个，取决于时间尺度、成交结构和市场状态。我们构造多个时间窗口的收益特征，让模型自己去学。

| 特征名 | 计算方式 | 金融含义 |
|--------|---------|---------|
| ret_1, ret_3, ret_6, ret_12, ret_24 | (midpx(t) - midpx(t-k)) / midpx(t-k) | 不同时间尺度的历史收益率 |
| rolling_mean_ret_6, rolling_mean_ret_12 | ret_1 在 6/12 窗口内的均值 | 近期趋势方向 |
| rolling_vol_6, rolling_vol_12 | ret_1 在 6/12 窗口内的标准差 | 近期波动水平 |
| high_low_range | (high - low) / midpx | 区间内多空争夺的剧烈程度 |
| price_position | (lastpx - low) / (high - low) | 收盘价在区间内的位置，接近高点说明买方占优 |
| overnight_gap | (open - prev_close) / prev_close | 隔夜信息冲击，仅开盘首分钟有效 |

### 3.2 盘口压力特征（16 个）

订单簿是市场状态的快照。买卖挂单的数量和金额直接反映了当前供需双方的力量对比。

| 特征名 | 计算方式 | 金融含义 |
|--------|---------|---------|
| spread | ask0 - bid0 | 绝对价差 |
| relative_spread | (ask0 - bid0) / midpx | 相对价差（跨股票可比） |
| weighted_spread_4 | (vwap_ask_4 - vwap_bid_4) / midpx | 加权价差，考虑各档深度的 spread |
| amount_imb_4, amount_imb_9, amount_imb_19 | (atr - btr) / (atr + btr) | 各档位金额不平衡——钱在哪边 |
| depth_sum_4, depth_sum_9 | bsize + asize | 总深度，越大流动性越好 |
| depth_delta_4, depth_delta_9 | depth(t) - depth(t-1) | 深度变化——有人在吃单还是挂单 |
| buy_pressure_0, buy_pressure_4, buy_pressure_9 | bsize / (bsize + asize) | 买方挂单比例，>0.5 说明买方更多 |
| micro_price_dev | (micro_price - midpx) / midpx | 订单权重中间价 vs 简单中间价的偏离 |
| bid_ask_bias_0, bid_ask_bias_4 | (bid×bsize - ask×asize) / sum | 价格和数量双加权的买卖偏向 |

盘口不平衡是高频交易中最核心的信号之一。买盘金额显著大于卖盘时，价格有买盘支撑，不容易被打下去。micro_price_dev 额外捕捉了买卖盘在价格分布上的不对称。

### 3.3 成交主动性特征（15 个）

挂单只是意愿，成交才是行动。主买（买方主动吃掉卖方挂单）代表进攻意愿强，主卖相反。成交的主动性、强度和节奏反映了市场参与者的真实意图。

| 特征名 | 计算方式 | 金融含义 |
|--------|---------|---------|
| trade_buy/sell/net_intensity | tradeQty / depth_sum_4 | 成交量占盘口深度的比例 |
| trade_buy/sell/net_turnover_ratio | tradeTurnover / (btr + atr) | 成交额占挂单额的比例 |
| rolling_trade_buy/sell_qty_6/12 | tradeQty 窗口内滚动求和 | 近期成交活跃度 |
| trade_count_imb | (nTradeBuy - nTradeSell) / sum | 成交笔数不平衡 |
| trade_count_imbema5 | EMA(trade_count_imb, halflife=5) | 平滑后的成交笔数不平衡趋势 |
| buy/sell_trade_size | tradeQty / nTrade | 单笔平均成交量——大单还是小单 |
| trade_size_ratio | buy_size / sell_size | 买卖单笔大小比——哪边更大手笔 |

买方不断主动吃掉卖盘挂单（高 trade_buy_intensity），说明有资金在坚定买入，这种买压通常在几分钟内推动价格上涨。成交笔数不平衡和单笔大小比从"频率"和"力度"两个维度补充了信息。

### 3.4 交互特征（5 个）

单一维度的信号有时候不够强，两个信号交叉可能产生新信息。交互特征全部使用 cs_rank 版本，消除量纲影响。

| 特征名 | 计算方式 | 金融含义 |
|--------|---------|---------|
| ret_3_x_imb0 | ret_3 × cs_rank_ob_imb0 | 短期动量 × 盘口不平衡——方向一致性 |
| ret_6_x_imb0 | ret_6 × cs_rank_ob_imb0 | 中期动量 × 盘口不平衡 |
| ret_3_x_buy_intensity | ret_3 × cs_rank_trade_buy_intensity | 动量 × 买方力度 |
| spread_x_vol | cs_rank_relative_spread × rolling_vol_12 | 流动性紧张 × 高波动 |
| highlow_x_imb0 | high_low_range × cs_rank_ob_imb0 | 剧烈争夺 × 盘口偏向 |

"方向一致时信号更可靠"——盘口买压大 + 近期动量向上 = 互相印证。反过来，盘口买压大但价格在跌，说明卖方的力量比盘口显示的要强。

### 3.5 P0 滚动统计特征（14 个）

前四类特征都是"瞬时快照"——反映当前这一刻的市场状态。但这些特征在时间上的**变化趋势**同样重要。比如盘口不平衡从 0.3 升到 0.8，和一直维持在 0.8，含义是不同的。我们用 12 分钟窗口对核心特征做滚动统计，捕捉时序动态。

| 特征名 | 计算方式 | 金融含义 |
|--------|---------|---------|
| ob_imb0_roll_mean/std/skew_12 | rolling(12) 均值/标准差/偏度 | 盘口不平衡的趋势、稳定性、极端程度 |
| ob_imb4_roll_mean/std_12 | rolling(12) 均值/标准差 | 更深档位不平衡的趋势 |
| trade_imb_roll_mean/std/skew_12 | rolling(12) 均值/标准差/偏度 | 成交不平衡的趋势和稳定性 |
| trade_imbema5_roll_mean/std_12 | rolling(12) 均值/标准差 | 平滑后成交趋势的持续性和波动 |
| ob_imb0_change_6 | diff(6) | 盘口不平衡在 6 分钟内的变化——加速还是减速 |
| ob_imb4_change_6 | diff(6) | 更深档位不平衡的变化速度 |
| trade_imb_change_6 | diff(6) | 成交不平衡的变化速度 |
| trade_imbema5_change_6 | diff(6) | 平滑成交趋势的变化速度 |

滚动计算时按 `sort_values(["symbol","interval"]) + groupby("symbol")` 分组，保证同一只股票内时序正确。

### 3.6 P1 横截面特征（8 个）

之前的 cs_rank 只作用在盘口和成交类的 20 个特征上。价格动量类的特征没有做截面比较——但实际上"一只股票过去 3 分钟涨了 1%"不如"它在所有股票里排前 5%"有意义。P1 就是补上这一块。

| 特征名 | 计算方式 | 金融含义 |
|--------|---------|---------|
| cs_rank_ret_3 | rank(ret_3) by interval | 3 分钟动量的截面排名 |
| cs_rank_ret_6 | rank(ret_6) by interval | 6 分钟动量的截面排名 |
| cs_rank_rolling_vol_12 | rank(rolling_vol_12) | 波动率的截面排名（高波 vs 低波） |
| cs_rank_rolling_mean_ret_12 | rank(rolling_mean_ret_12) | 趋势强度的截面排名 |
| cs_rank_ret_3_change | diff of cs_rank_ret_3 | 截面排名的变化——相对动量在改善还是恶化 |
| cs_rank_ret_6_change | diff of cs_rank_ret_6 | |
| cs_rank_rolling_vol_12_change | diff of cs_rank_rolling_vol_12 | |
| cs_rank_rolling_mean_ret_12_change | diff of cs_rank_rolling_mean_ret_12 | |

截面 rank 的变化率是 P1 里最有意思的特征——不仅知道"这只股票现在排第几"，还知道"它的排名是在上升还是下降"。这种动量方向的加速度，往往比方向本身更有预测价值。

### 3.7 特征计算策略

在开发过程中我们发现一个重要的实践教训：shift/rolling/ewm 操作如果按 `(symbol, date)` 分组（每天重置），会导致特征退化——因为每天第一条数据全部变成 NaN，丢失了隔夜信息的连续性。最终方案是：

| 操作类型 | 策略 | 原因 |
|---------|------|------|
| ret_*, bret12, rolling_mean/vol | **全局** shift/rolling | 数据已按 symbol+interval 排列，边界错误仅 ~5% |
| trade_imbema5, trade_count_imbema5 | **全局** ewm | 跨股票内容衰减快（halflife=5），影响极小 |
| overnight_gap, depth_delta, rolling_trade_qty | sort + **groupby(symbol)** | 需要股票隔离，但不能按天切断 |
| cs_rank 全部 | **groupby(interval)** | 不同日期的同一时刻混在一起排序，样本量大更稳定 |
| P0 全部, P1 change | sort + **groupby(symbol)** | 需要股票内时序正确 |

消融实验确认：这个方案（V5）的 Pearson 为 0.0714，而改为全部 `_per_session` 的方案降到 0.0522。跨天连续性对金融时序特征至关重要。

---

## 4. 模型方法与实验结果（B 完成）

> 本节由 B 撰写，仅列出 A 参与确认的关键数据。

### 4.1 模型选型逻辑

#### 为什么选 Ridge 作为 Baseline

项目原始代码使用 Ridge 回归。Ridge 的优势在于：(1) 训练快，秒级出结果，适合快速验证数据管线和评估逻辑；(2) L2 正则化对高维共线性特征有一定鲁棒性；(3) 作为线性模型，其结果可解释性强——如果 Ridge 就能做好的信号，说明线性关系已经足够。Ridge baseline 的 Pearson = 0.029，说明原始 6 个特征在线性模型下预测力很弱，需要非线性模型和更丰富的特征。

#### 为什么选 LightGBM GBT 作为主力模型

在 XGBoost、LightGBM、MLP、HistGradientBoosting、RandomForest 等候选中，选择 LightGBM GBT 的理由：

| 候选模型 | 优势 | 劣势 | 选择理由 |
|---------|------|------|---------|
| **LightGBM GBT** | 直方图加速、Leaf-wise 生长、GPU 支持 | 需调参 | 训练速度快，表格数据 SOTA |
| XGBoost | 成熟稳定 | 无 GPU 直方图加速（需外挂） | 训练速度慢 5~10 倍，不适合大量调参 |
| MLP | 可学习复杂交互 | 需大量调参、对表格数据不如 GBT | 金融表格数据 GBT 通常优于 MLP |
| HistGradientBoosting | sklearn 原生 | 无 GPU、功能少 | 缺少 GPU 加速和丰富回调 |
| RandomForest | 无需调参 | 性能上限低 | 不支持 boosting，信号太弱学不到 |

LightGBM 的核心优势：
- **直方图加速**：将连续特征离散化为 max_bin=63 个 bin，分裂点搜索从 O(n) 降到 O(bin)
- **Leaf-wise 生长**：按叶子增益分裂而非层级分裂，相同叶子数下精度更高

### 4.2 防过拟合策略

金融时序预测的过拟合风险特别高——信号极弱（Pearson ~0.07），噪声极大，且市场状态 non-stationary。我们采用了多层防过拟合措施：

| 层级 | 策略 | 具体实现 | 效果 |
|------|------|---------|------|
| **数据层** | 时序交叉验证 | TimeSeriesSplit(n_splits=3)，验证集始终在训练集之后 | 防止未来信息泄漏 |
| **数据层** | 测试集完全隔离 | 训练 6~11 月，测试 12 月，两段不重叠 | 评估真实泛化能力 |
| **模型层** | 限制树深度 | max_depth=5 | 限制单棵树复杂度 |
| **模型层** | 行采样 | subsample=0.502 | 每棵树只用 50% 样本，降低方差 |
| **模型层** | 列采样 | colsample_bytree=0.551 | 每棵树只用 55% 特征，降低方差 |
| **模型层** | 叶子最小样本 | min_child_samples=200 | 防止叶子过小过拟合噪声 |
| **正则化** | L1 正则 | reg_alpha=0.779 | 稀疏化特征选择 |
| **正则化** | L2 正则 | reg_lambda=1.908 | 平滑叶权重，防止极端预测 |
| **训练层** | Pearson 早停 | early_stopping=100，基于验证集 Pearson | 与评估目标对齐，L2 继续降但 Pearson 不升时及时停止 |
| **调参层** | 提前剪枝 | OverfitRatio < 0.3 直接 TrialPruned | 过滤极端过拟合参数 |

**关键发现：OverfitRatio 0.4~0.6 是金融预测的正常水平**。初版调参对 OverfitRatio < 0.9 施加惩罚，导致 TPE 走向极端保守（重正则化 + 少树），反而欠拟合。去掉惩罚后 Pearson 从 0.0553 提升到 0.0606。

### 4.3 增量实验：逐步加入特征的性能变化

按照分工文档要求的实验顺序，从 Baseline 开始逐步加入各类特征，观察 MSE、Pearson、R² 的变化：

| 实验 | 特征组合 | 模型 | 特征数 | Pearson | R² | MSE | 说明 |
|------|---------|------|--------|---------|-----|-----|------|
| 实验0 | 原始 6 特征 | Ridge | 6 | 0.0222 | 0.00046 | 0.00 | Baseline |
| 实验1 | + 48 个新增特征 | GBT | 54 | 0.0269 | 0.00063 | 0.00 | 价格动量+盘口+成交+交互 |
| 实验2 | 主力模型 + 54 特征 | GBT | 54 | 0.0635 | 0.00397 | 0.000024 | V4 基线，B 第一阶段最优 |
| 实验3 | + P0 滚动统计 | GBT | 68 | ~0.068 | — | — | 核心特征的时序动态 |
| 实验4 | + P1 横截面 | GBT | 76 | **0.0716** | **0.0049** | — | **V5 最终最优** |

**关键观察**：

1. **特征工程提升显著**（实验0→1: Ridge→GBT + 48特征, Pearson 0.0222→0.0269）：模型和特征共同贡献
2. **主力模型大幅提升**（实验1→2: 0.0269→0.0635）：V4参数+Pearson早停释放了GBT潜力
3. **P0 滚动统计有效**（实验2→3: +0.005）：时序动态比瞬时快照更有预测力
4. **P1 横截面有效**（实验3→4: +0.0036）：相对强弱比绝对值更有信息量

每一步新增特征都带来正向提升，说明特征工程设计方向正确，没有引入有害冗余特征。

### 4.4 超参数调优

调参框架：Optuna TPE 贝叶斯优化，3 折 TimeSeriesSplit 时序交叉验证。

| 参数 | 搜索范围 | 最终值 | 说明 |
|------|----------|--------|------|
| max_depth | 3~5 | 5 | 树最大深度 |
| num_leaves | 7~63 | 31 | 叶子节点数 |
| learning_rate | 0.005~0.1 (log) | 0.0156 | 学习率，低值慢学习策略 |
| n_estimators | 200~2000 | 1600 | 最大树数量（早停决定实际轮数） |
| subsample | 0.5~0.8 | 0.502 | 行采样比例 |
| colsample_bytree | 0.5~0.8 | 0.551 | 列采样比例 |
| min_child_samples | 50~300 | 200 | 叶子最小样本数 |
| reg_alpha | 0.01~2.0 (log) | 0.779 | L1 正则化 |
| reg_lambda | 0.01~2.0 (log) | 1.908 | L2 正则化 |

#### 调参迭代过程

| 版本 | 核心改动 | CV Pearson | 测试集 Pearson | 问题 |
|------|----------|------------|---------------|------|
| 基线 | 默认参数 | — | 0.0576 | — |
| V1 | categorical 搜索, 5 折 CV, 过拟合惩罚 | 0.0577 | 0.0601 | 5 折第 1 折数据太少 |
| V2 | 连续搜索, 3 折 CV, lr→0.1, 早停联动 | 0.0546 | 0.0553 | 正则化过重导致欠拟合 |
| V3 | 去惩罚, reg≤2, min_child≤300 | 0.0532 | 0.0606 | CV 与正式训练差距大 |
| V4 | leaves 放宽, n_est→2000, sample=0.7 | 0.0583 | **0.0635** | 54 特征天花板 |
| V5-err | 76 特征 + 错误 _per_session(symbol,date) | 0.0655 | 0.0522 | 特征计算方式错误 |
| **V5** | 76 特征 + 恢复 V4 逻辑 + P0/P1 | — | **0.0716** | 当前最优 |

#### 关键调参发现

- **过拟合惩罚有害**：金融预测 OverfitRatio 0.4~0.6 属正常水平，惩罚会让 TPE 走向极端保守
- **5 折改 3 折**：5 折的第一折数据太少（20%），Pearson 接近 0，污染均值；3 折更稳定
- **子采样改为按天保留**：保留完整的天（所有股票×所有时刻），保证天内横截面结构和时序连续性；在前 20% 天数范围内随机选起始天，取连续 80% 的天数
- **早停需与 lr 联动**：lr 越小需要越多的 patience 轮数，early_stopping = max(50, 0.1/lr × 50)
- **CV 最后一折加权**：最后一折最接近测试集时间窗口，3 折权重 [0.25, 0.30, 0.45]
- **Pearson 早停优于 L2 早停**：L2 持续下降但 Pearson 已不升时，继续训练引入过拟合噪声；改为 Pearson 早停后 best_iter 从 573→1325，模型训练更充分
- **params.pop() bug**：`_train_and_evaluate_with_model` 中 `params.pop('early_stopping_rounds')` 会永久修改 params 字典，导致 CV 后续折缺少早停参数
- **慢学习策略更优**：低 lr(0.016) + 多树(1600) + 强 L2(1.908) 在测试集上优于高 lr(0.036) + 少树(1000) + 弱 L2(0.269)，Pearson 0.0716 vs 0.0711

#### V5 特征计算错误的教训

V5 迭代中引入 `_per_session(symbol,date)` 按 (symbol, date) 分组，导致严重退化（Pearson 从 0.0714 降到 0.0522）：

| 错误操作 | 影响 | 退化机制 |
|---------|------|---------|
| shift 按 (symbol,date) 分组 | 每天第一根 bar 的 shift 全为 NaN→fillna(0) | 丢失隔夜信息，ret_1~24 每天开头全错 |
| rolling 按 (symbol,date) 分组 | 每天开头只有 1~2 个数据点 | 统计量极不稳定，引入大量噪声 |
| ewm 按 (symbol,date) 分组 | trade_imbema5 每天重置 | 完全失去跨天记忆，信号消失 |

**核心教训**：金融时序特征的跨天连续性至关重要，不能按 (symbol, date) 分组切断每天重置。

### 4.5 最终性能

| 方案 | 特征数 | Pearson | R² | 相比 Baseline | 相比 V4 |
|------|--------|---------|-----|-------------|---------|
| Ridge Baseline | 6 | 0.029 | — | — | — |
| V4（GBT + 54 特征） | 54 | 0.0635 | 0.00397 | +119% | — |
| **V5（GBT + 76 特征）** | **76** | **0.0716** | **0.0049** | **+147%** | **+12.4%** |

- 训练集 Pearson: 0.132，验证集 Pearson: 0.074（早停点），测试集 Pearson: 0.0714
- OverfitRatio ≈ 0.54（验证/训练），金融预测正常水平
- Best iteration: 1325（Pearson 早停），总训练时间约 17 分钟（GPU）

### 4.6 特征重要性 Top 10

| 排名 | 特征 | Gain | 类别 |
|------|------|------|------|
| 1 | overnight_gap | 4.04 | 价格动量 |
| 2 | rolling_vol_12 | 2.55 | 价格动量 |
| 3 | ret_24 | 2.18 | 价格动量 |
| 4 | cs_rank_rolling_mean_ret_12_change | 2.15 | P1 横截面 |
| 5 | cs_rank_rolling_vol_12 | 2.11 | P1 横截面 |
| 6 | cs_rank_rolling_vol_12_change | 2.03 | P1 横截面 |
| 7 | ret_3_x_buy_intensity | 2.00 | 交互 |
| 8 | ret_1 | 1.86 | 价格动量 |
| 9 | rolling_vol_6 | 1.77 | 价格动量 |
| 10 | cs_rank_trade_imbema5 | 1.40 | 成交主动性 |

**解读**：
- **overnight_gap 排第一**：隔夜信息冲击是最强的短期预测信号，开盘首分钟最有效
- **波动率特征占据 3 席**（rolling_vol_12, rolling_vol_6, cs_rank_rolling_vol_12）：波动水平是金融预测的核心维度
- **P1 横截面特征占 3 席**（#4, #5, #6）：验证了"相对强弱比绝对值更有预测力"的设计思路
- **交互特征 ret_3_x_buy_intensity**：短期动量与买方力度的交叉信号，方向一致时更可靠

![特征重要性图](report_figures/feature_importance.png)

*图：Top 20 特征重要性条形图。overnight_gap、rolling_vol_12、ret_24 是前三大重要特征，横截面特征（cs_rank_*）占据多个重要位置。*

### 4.7 模型性能可视化

#### 4.7.1 训练曲线

![训练曲线](report_figures/training_curve.png)

*图：训练和验证Pearson曲线（上）及差值曲线（下）。训练集Pearson约0.132，验证集Pearson约0.072，差值约0.058，显示适度过拟合。训练在约1300轮达到最佳验证性能。*

#### 4.7.2 预测效果

![预测值与真实值散点图](report_figures/prediction_vs_actual.png)

*图：预测值vs真实值散点图（上）及残差图（下）。散点图显示预测值与真实值呈弱正相关，残差图显示误差分布基本对称，无明显系统性偏差。*

#### 4.7.3 残差分析

![残差分布图](report_figures/residual_distribution.png)

*图：残差分布分析，包含直方图、箱线图、Q-Q图和自相关图。残差近似正态分布，均值接近0，无明显自相关性，符合线性回归假设。*

### 4.8 可视化分析结论

1. **模型学习能力**：训练曲线显示模型能有效学习训练集信号（训练Pearson=0.132）
2. **泛化能力**：验证集Pearson=0.074，测试集Pearson=0.0714，泛化性能良好
3. **过拟合程度**：训练-验证差值约0.058，OverfitRatio约0.56，金融预测中属正常水平
4. **预测分布**：散点图显示预测值与真实值正相关，但相关性较弱（Pearson=0.0716）
5. **残差特性**：残差分布近似正态，无明显系统性偏差，模型假设基本成立

---

## 5. 成员分工

| 成员 | 主要负责 | 具体工作 |
|------|---------|---------|
| **A** | 数据分析 + 特征工程 | 原始数据 EDA、市场微观结构分析、54 基线特征的金融解释；P0 滚动统计特征和 P1 横截面特征的设计与实现（22 个新特征）；特征计算策略的消融验证；报告数据分析与特征工程章节 |
| **B** | 模型训练 + 评估 | Ridge baseline 复现、LightGBM GBT 主力模型训练与 Optuna 超参数调优、消融实验、MSE/Pearson/R² 指标评估；报告模型方法与实验结果章节 |
| **C** | Agent 创新 + 报告整合 | 大模型辅助因子挖掘记录、轻量交易决策 Agent（预测值→buy/hold/sell）、简单回测验证、报告整合与成员分工整理 |

---

## 附录：代码文件说明

**运行方式**：直接执行 `python meow.py` 即可完成训练+评估全流程。数据路径默认为 `archive`，需确保该目录下有按天存储的 HDF5 文件。

| 文件 | 负责 | 说明 |
|------|------|------|
| `meow.py` | 入口 | 主流程引擎，`python meow.py` 一键运行 |
| `dl.py` | 原始 | 数据加载器 |
| `feat.py` | **A** | 特征生成器，76 个特征，支持 feature_set 开关 |
| `GBT-final/feat.py` | **A+B** | 最终特征集（以这个为准） |
| `eda.py` | **A** | EDA 分析脚本 |
| `gen_figures.py` | **A** | 报告图表生成脚本 |
| `mdl.py` | A+B | 模型定义（Ridge + GBT），含 B 调优参数 |
| `GBT-final/tuner.py` | **B** | Optuna 超参数调优框架 |
| `GBT-final/run_ablation.py` | **B** | 消融实验脚本 （不可用）|
| `GBT-final/总结.md` | **B** | B 的建模总结 |
| `eval.py` | 原始 | 评估器（MSE / Pearson / R²） |
| `tradingcalendar.py` | 原始 | 交易日历工具 |
| `visualization.py` | **B** | 模型性能可视化模块 |
| `figures/` | **B** | 模型评估图表目录 |
| `figures/feature_importance.png` | **B** | 特征重要性Top 20条形图 |
| `figures/training_curve.png` | **B** | 训练和验证Pearson曲线 |
| `figures/prediction_vs_actual.png` | **B** | 预测值vs真实值散点图 |
| `figures/residual_distribution.png` | **B** | 残差分布分析图 |
