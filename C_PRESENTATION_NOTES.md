# C 汇报速记

## 30 秒定位

我负责的是 Agent 创新和实验整合。项目主线仍然是课程要求的 `fret12` 预测，也就是预测股票 12 分钟后的收益率。C 没有把项目包装成完整自动交易系统，而是在 B 的预测模型之后加了一个轻量 Trading Agent 决策层，检查 `forecast` 能不能转成有方向的 `buy / hold / sell` 动作。

核心链路是：

```text
市场状态 -> 76 个特征 -> B 的 GBT/LightGBM 模型 -> forecast -> Trading Agent -> buy / hold / sell -> simplified reward
```

## 2 分钟汇报稿

我这部分主要做了两件事。

第一是 Agent-assisted alpha mining。我们不是让 Agent 直接替代建模，而是让它作为研究助手，结合字段含义、市场微观结构和 A/B 的代码，把 76 个特征整理成七类可解释因子：盘口压力、流动性和价差、成交主动性、动量反转、P0 时序状态、P1 横截面相对强弱，以及交互因子。最后特征是否有效，仍然由验证集上的 Pearson、R2 和 MSE 决定。

第二是 lightweight Trading Agent。B 的模型输出 `forecast` 后，Agent 不再重新预测价格，而是把这个 alpha signal 转成三个动作：`1` 做多，`0` 空仓，`-1` 做空。默认策略是每个 `date + interval` 横截面里，做多 forecast 最高的 Top 10%，做空 forecast 最低的 Bottom 10%，其余持有空仓，并扣除 1 bps 的简化交易成本。

真实数据复现里，B 最终模型在 2023 年 12 月测试集上的 Pearson 是 `0.0677`，R2 是 `0.00443`。接入 C 的 top-bottom 10% Agent 后，Top 10% 股票未来 12 分钟平均收益是 `0.00043309`，Bottom 10% 是 `-0.00034123`，多空差是 `0.00077432`，约等于 `7.74 bps`。

这个结果说明，B 的 forecast 不只是让 Pearson 变高，也能在横截面上形成可解释的交易方向。进一步看 decile，底部十分位平均真实收益是 `-3.41 bps`，顶部十分位是 `4.33 bps`，从低到高整体递进。策略敏感性也符合直觉：top-bottom 5% 的多空差最大，约 `10.42 bps`；扩大到 10% 是 `7.74 bps`；扩大到 20% 后下降到 `5.51 bps`，说明越极端的预测分组信号越强。

所以我的结论是：这不是完整 Agent trading 系统，而是 Agent-assisted alpha mining 加轻量交易决策原型。它的价值是把课程主任务从“预测指标更好”推进到“预测信号是否有交易方向价值”。

## 必背数字

```text
Dataset: Kaggle MEOW dataset v3, 144 个 h5 文件
Train: 2023-06-01 到 2023-11-30
Test: 2023-12-01 到 2023-12-29

A root model Pearson: 0.0511
B GBT-final Pearson: 0.0677
B GBT-final R2: 0.00443

C default policy: top-bottom 10%, cost 1 bps
samples: 1,462,884
trade_ratio: 0.1986
mean_net_return: 0.00005713
trade_mean_net_return: 0.00028772
hit_rate: 0.5008
long_mean_target: 0.00043309
short_mean_target: -0.00034123
long_short_spread: 0.00077432, about 7.74 bps

Period-level check:
period_count: 4,746
period_long_short_spread: 0.00077266, about 7.73 bps
positive_spread_period_rate: 0.7219
```

## 老师可能追问

**这算 Agent trading 吗？**

严格说不算完整 Agent trading。完整系统要有真实撮合、滑点、仓位管理、风险控制和订单生命周期。我们做的是预测信号之后的轻量 Agent 决策层：输入市场状态和模型 forecast，输出 buy/hold/sell，并用真实 `fret12` 做简化 reward 验证。

**为什么用 top-bottom 策略？**

因为这个任务本质上是横截面股票预测。top-bottom 能直接检验 forecast 的排序能力：预测最高的一组未来收益是否更高，预测最低的一组是否更低。它比固定阈值更适合多股票同一时刻比较。

**为什么不是只看 Pearson？**

Pearson 说明预测值和真实收益有线性相关，但交易上还要看信号能否转成方向动作。所以 C 增加了 long/short/hold 决策和多空收益差，检查信号可交易性。

**为什么不展示真实资金曲线？**

因为当前模块是 signal backtest，不模拟真实成交和连续持仓。为了避免过度承诺，我只展示样本级和横截面期级的收益差、命中率、decile 递进关系，不把它说成真实账户收益。

**老师质疑逐行样本均值不稳怎么办？**

可以补一句：我也做了 `date + interval` 级别的聚合检查。测试集有 `4,746` 个横截面期，默认 top-bottom 10% 策略的期级平均多空差约 `7.73 bps`，并且 `72.19%` 的横截面期多空差为正，所以不是只靠少数样本堆出来的。

**交易成本怎么处理？**

当前用简化成本 `cost_bps / 10000 * abs(action)`，默认每个交易动作扣 1 bps。它不能覆盖真实滑点和冲击成本，但能避免完全无成本的理想化。

**为什么 top-bottom 5% 比 20% 好？**

这符合排序信号的特征。越靠近预测分布两端，forecast 越极端，信号越强；扩大交易范围后会纳入更多弱信号，收益差被稀释。

## PPT 建议

1. 一页讲定位：课程主线是 `fret12` 预测，C 是预测后的轻量 Agent 决策层。
2. 一页讲方法：`forecast -> rank -> top/bottom -> action -> reward`。
3. 一页贴 decile 图：底部十分位 `-3.41 bps`，顶部十分位 `4.33 bps`。
4. 一页贴策略敏感性：5%、10%、20% 多空差递减。
5. 最后一页讲边界：不是完整交易仿真，价值在检验 alpha signal 的方向性。

可直接使用的图在：

```text
outputs/C_decile_mean_target.svg
outputs/C_policy_spread.svg
```
