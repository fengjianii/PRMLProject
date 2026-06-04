# C 部分报告材料：Agent 辅助因子挖掘与轻量交易决策原型

## 1. C 模块定位

本项目的主任务是预测股票 12 分钟后的 forward return，即 `fret12`。因此，我们没有把系统包装成完整的自动交易系统，而是在完成预测主线后，额外加入一个轻量 Trading Agent 决策层，用来验证模型输出的 `forecast` 是否能够转化为有方向性的交易动作。

完整关系可以理解为：

```text
市场数据 -> 76 个特征 -> GBT 预测模型 -> forecast -> Trading Agent -> buy / hold / sell -> simplified reward
```

其中，A 负责把市场状态编码成特征，B 负责训练预测模型和评价 MSE / Pearson / R2，C 负责把预测信号接成一个轻量交易决策闭环。

B 最新一轮调参后，`GBT-final` 中的最佳模型 Pearson 达到 0.0655，相比早期 baseline 的 0.0576 有明显提升。因此 C 的 Agent 回测优先使用 B 最终模型导出的 `forecast`，而不是早期根目录模型的预测结果。

## 2. Agent 辅助因子挖掘

Agent 在本项目中不是直接替代建模，也不是直接下单，而是作为研究助手参与因子挖掘。它的输入包括课程文档里的字段定义、市场微观结构解释、A/B 已实现的特征代码和实验结果。它的输出是候选因子方向和解释框架。

最终特征是否采用，不由 Agent 直接决定，而是由验证集上的 MSE、Pearson Correlation 和 R2 决定。这样可以避免把大模型建议当成结论，而是把它作为生成候选假设的工具。

当前 76 个特征可以归纳为七类。需要注意的是，B 在 `GBT-final` 中保留了 feature set 开关，可以比较 `v4`、`no_p1`、`no_p0` 和 `full`，这为 C 后续解释 Agent 信号质量提供了一个自然切入点：如果 full 模型的 forecast 在 top-bottom 回测中表现更好，说明 P0/P1 新特征不只提升预测指标，也提升信号的交易方向价值。

第一类是盘口压力因子。它们描述买卖盘深度、金额和挂单压力的相对强弱，例如 `cs_rank_ob_imb0`、`cs_rank_ob_imb4`、`cs_rank_buy_pressure_0`、`cs_rank_amount_imb_4` 等。它们背后的假设是：如果某只股票在同一时刻的买盘压力相对更强，那么短期价格更可能获得支撑。

第二类是流动性和价差因子，包括 `cs_rank_spread`、`cs_rank_relative_spread`、`cs_rank_weighted_spread_4`、`cs_rank_depth_sum_4` 等。价差和盘口深度决定了短期交易摩擦，也影响预测信号能否转化成真实交易动作。

第三类是成交主动性因子，包括 `cs_rank_trade_buy_intensity`、`trade_sell_intensity`、`cs_rank_trade_net_intensity`、`trade_count_imbema5` 等。成交比挂单更接近真实行动，主动买入或主动卖出的强度能够反映短期资金方向。

第四类是动量和反转因子，包括 `ret_1`、`ret_3`、`ret_6`、`ret_12`、`rolling_mean_ret_12`、`rolling_vol_12` 等。短周期价格变化可能延续，也可能回撤；不同窗口可以帮助模型判断当前行情更像动量还是反转。

第五类是 P0 时序状态因子，例如 `ob_imb0_roll_mean_12`、`trade_imb_roll_std_12`、`buy_intensity_change_6`。它们不只看当前值，还看盘口压力和成交强度在过去一段时间里是在增强还是减弱。

第六类是 P1 横截面相对强弱因子，例如 `cs_rank_ret_3`、`cs_rank_ret_12`、`cs_rank_rolling_vol_12`、`cs_rank_ret_3_change`。这类特征强调“相对位置”：不是只看一只股票涨了多少，而是看它在同一时刻所有股票中的排名。

第七类是交互因子，例如 `ret_3_x_imb0`、`ret_6_x_imb0`、`ret_3_x_buy_intensity`、`spread_x_vol`。它们用于表达多个信号同时出现时的状态，例如价格短期上涨且盘口买压增强时，信号可能比单独看动量更可靠。

## 3. Trading Agent 设计

Trading Agent 的输入不是原始订单，而是 B 的模型输出 `forecast`。这个 `forecast` 已经吸收了 A 构造的 76 个市场状态特征，因此它可以被视为一个 alpha signal。

Agent 的动作空间定义为：

```text
1  = buy / long
0  = hold / no trade
-1 = sell / short
```

当前实现了两种策略。

第一种是横截面 top-bottom policy。每个 `date + interval` 下，把所有股票按 `forecast` 排序，买入预测值最高的 Top 10%，卖出预测值最低的 Bottom 10%，其余股票保持空仓。这个策略适合验证模型预测值在横截面上是否具有排序能力。

第二种是 threshold policy。它根据固定阈值决定动作：当 `forecast` 高于买入阈值时买入，低于卖出阈值时卖出，中间保持空仓。这个策略更接近单资产信号触发逻辑。

## 4. Reward 与回测方式

当前回测是轻量 signal backtest，不是完整交易所仿真。我们使用真实 `fret12` 作为 12 分钟后的简化收益，并加入可选交易成本。

核心公式是：

```text
gross_return = action * fret12
net_return = gross_return - transaction_cost
```

其中 `transaction_cost = cost_bps / 10000 * abs(action)`。如果设置 `cost_bps = 1`，表示每次交易扣除 1 bps 的简化成本。

回测输出包括样本数、做多次数、做空次数、空仓次数、交易比例、平均毛收益、平均净收益、交易命中率、做多组真实平均收益、做空组真实平均收益，以及多空组真实收益差。

## 5. 局限性

这个模块不模拟真实交易撮合，也不处理订单排队、限价单成交概率、滑点、撤单、仓位约束和多期持仓重叠问题。因此它不能被解释为完整的 Agent trading 系统。

它的价值在于补上预测任务之后的一步：如果模型只是在 MSE / Pearson / R2 上表现更好，但预测值无法转化为方向性交易动作，那么它的交易意义有限。轻量 Trading Agent 可以帮助我们从“预测准确性”进一步检查“信号可交易性”。

## 6. 汇报口径

汇报时可以这样说：

```text
我负责 Agent 创新和实验整合部分。我们的项目主线仍然是课程要求的 12 分钟收益率预测，没有把它包装成完整自动交易系统。在主线之外，我做了两件事：第一，用 Agent 辅助因子挖掘，把 76 个特征组织成盘口压力、流动性、成交主动性、动量反转、时序状态、横截面相对强弱和交互因子几类；第二，在模型输出 forecast 后加入轻量 Trading Agent，把预测值转成 buy / hold / sell 动作，并用真实 fret12 做简化 reward 验证预测信号是否有交易方向价值。
```

如果老师追问为什么不做完整 Agent trading，可以回答：

```text
因为课程评价核心是 fret12 预测，完整交易系统还需要真实撮合、滑点、交易成本、仓位管理和风险控制。我们这次只做预测信号后的轻量决策层，目的是验证 alpha signal，而不是模拟完整交易所。
```
