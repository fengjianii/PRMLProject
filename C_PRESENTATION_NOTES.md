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

我后面又把这部分做成了自动化工具。`agent_factor_mining.py` 可以从 h5 数据自动生成特征，对每个因子计算 Pearson、Spearman、横截面 IC、ICIR 和 top-bottom 分组收益差，然后自动输出因子报告和 factor-aware Agent 配置。

第二是 lightweight Trading Agent。B 的模型输出 `forecast` 后，Agent 不再重新预测价格，而是把这个 alpha signal 转成三个动作：`1` 做多，`0` 空仓，`-1` 做空。默认策略是每个 `date + interval` 横截面里，做多 forecast 最高的 Top 10%，做空 forecast 最低的 Bottom 10%，其余持有空仓，并扣除 1 bps 的简化交易成本。

真实数据复现里，B 最终模型在 2023 年 12 月测试集上的 Pearson 是 `0.0677`，R2 是 `0.00443`。接入 C 的 top-bottom 10% Agent 后，Top 10% 股票未来 12 分钟平均收益是 `0.00043309`，Bottom 10% 是 `-0.00034123`，多空差是 `0.00077432`，约等于 `7.74 bps`。

这个结果说明，B 的 forecast 不只是让 Pearson 变高，也能在横截面上形成可解释的交易方向。进一步看 decile，底部十分位平均真实收益是 `-3.41 bps`，顶部十分位是 `4.33 bps`，从低到高整体递进。策略敏感性也符合直觉：top-bottom 5% 的多空差最大，约 `10.42 bps`；扩大到 10% 是 `7.74 bps`；扩大到 20% 后下降到 `5.51 bps`，说明越极端的预测分组信号越强。

所以我的结论是：这不是完整 Agent trading 系统，而是 Agent-assisted alpha mining 加轻量交易决策原型。它的价值是把课程主任务从“预测指标更好”推进到“预测信号是否有交易方向价值”。

如果时间允许，可以加一句自动因子挖掘：

> 我还做了一个自动因子挖掘 Agent。它在 12 月测试区间自动发现，短周期反转是最强的单因子方向，比如 `cs_rank_ret_3` 的 mean IC 是 `-0.049636`，说明横截面短期涨幅越靠前，未来 12 分钟越容易回撤。Agent 会把这类因子标记成 reverse / short / risk signal，并自动生成一个 factor-aware Agent 配置。这个配置的诊断多空差约 `4.88 bps`，主要用于提出下一轮候选策略。

如果老师对 Agent trading 感兴趣，可以继续补执行层：

> 我还加了一个简化执行仿真层。它会读取真实盘口，近似模拟 market order、limit order 和 hybrid order。market order 会穿越价差并加入滑点，limit order 会根据对手方成交量和排队量判断是否成交，同时加入每期仓位上限和资金曲线。结果显示，市价强成交虽然 100% 成交，但会被滑点吃掉；limit 模式成交率约 `66.28%`，但最终资金为 `10,669,876.40`，最大回撤只有 `-0.20%`。这说明 C 的 Agent 不只看预测方向，也开始考虑订单执行质量。

如果还要讲强化学习：

> 最后我做了一个轻量 Q-learning 策略选择器。它根据 forecast 离散度和波动状态，在 hold、top-bottom 5%、10%、20% 之间选择动作，然后用扣成本后的收益更新 Q 值。这个原型在 4,746 个横截面期上跑完，最终资金为 `10,268,736.34`，positive reward rate 为 `70.97%`。它不是深度强化学习，但补上了“收益反馈 -> 更新策略偏好”的闭环。

如果想把系统感讲得更完整，可以补最后一句：

> 我最后又加了一个 Agent Controller 总控层。它不重新训练模型，而是把预测回测、因子挖掘、执行仿真、RL selector 的输出统一读进来，自动生成 `C_AGENT_DASHBOARD.md` 和 buy / sell / hold 决策样例。这样 C 的工作不是零散脚本，而是一个从 forecast 到 action、从 action 到 reward、再到执行和策略更新的完整研究原型。

如果现场可以展示页面，可以接着说：

> 为了汇报更直观，我还做了一个静态 GUI。它是一个深色 Trading Agent Console，左侧是模块导航，主区域展示 signal、factor mining、execution、RL 和 decision tape。它不是额外改结果，只是把 C 的已有实验输出变成一个可以现场演示的界面。

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

Auto factor mining:
best factor: cs_rank_ret_3
cs_rank_ret_3 mean_ic: -0.049636
cs_rank_ret_3 spread: -4.45 bps
ret_1 spread: -5.63 bps
factor-aware Agent diagnostic spread: 4.88 bps

Execution simulation:
market final equity: 8,331,537.84
hybrid final equity: 9,778,965.38
limit fill_rate: 0.6628
limit final equity: 10,669,876.40
limit max_drawdown: -0.0020

RL policy:
steps: 4,746
final_equity: 10,268,736.34
positive_reward_rate: 0.7097
max_drawdown: -0.00038693

Agent Controller:
dashboard: outputs/C_AGENT_DASHBOARD.md
decision examples: outputs/C_agent_decision_examples.csv
auto summary: signal / factor / execution / RL

GUI:
page: outputs/C_AGENT_GUI.html
style: dark dashboard / left nav / KPI cards / charts / decision tape
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

**自动因子挖掘有什么用？**

它把 Agent 从“解释报告”变成“候选策略生成器”。脚本会自动算每个因子的 IC、ICIR 和分组收益差，判断方向是 positive 还是 reverse，再输出 `C_factor_agent_config.json`。这份配置可以作为下一轮模型消融、风控过滤器或 factor-aware policy 的起点。

**你说加了真实撮合，是不是完整交易所仿真？**

不是。现在是 execution approximation。我们用盘口快照和聚合成交量近似撮合：market order 穿越价差，limit order 根据对手方成交量和估计排队量决定成交。这比纯 signal backtest 更接近交易，但还不是逐笔订单级撮合。

**为什么 market 亏、limit 反而正？**

因为高频短周期里价差和滑点很重要。market order 为了保证成交要立刻吃价，收益会被执行成本吞掉；limit order 牺牲成交率，但拿到更好的入场价格，所以资金曲线更稳。这正好说明 Trading Agent 不能只判断方向，还要判断怎么执行。

**强化学习是不是很完整？**

不是完整深度强化学习，是轻量 Q-learning / contextual-bandit 原型。它的作用是演示 reward feedback：Agent 观察状态、选择策略、得到收益、更新 Q 值。我们把它作为未来做完整 RL trading 的雏形。

**Agent Controller 有什么意义？**

它的意义是把 C 的模块从“几个脚本”收束成一个可解释系统。Controller 读取已有实验输出，自动生成 Dashboard 和具体 buy / sell / hold 决策样例。汇报时可以说：我不只给出平均收益，还能展示某个横截面里 Agent 为什么买、为什么卖、为什么选择 hold。

**GUI 是不是又做了一个系统？**

不是。GUI 只是展示层，读取的是前面脚本已经生成的 CSV 和 Markdown 输出。它的作用是把实验链路变得可视化，方便现场解释 C 的 Agent 从预测信号走到交易动作、执行质量和策略更新。

## PPT 建议

1. 一页讲定位：课程主线是 `fret12` 预测，C 是预测后的轻量 Agent 决策层。
2. 一页讲方法：`forecast -> rank -> top/bottom -> action -> reward`。
3. 一页贴 decile 图：底部十分位 `-3.41 bps`，顶部十分位 `4.33 bps`。
4. 一页贴策略敏感性：5%、10%、20% 多空差递减。
5. 一页讲执行和 RL：market / limit / hybrid 对比，加上 Q-learning 闭环。
6. 最后一页讲 GUI / Controller 和边界：它能一键汇总结果和样例，但还不是生产级自动交易系统。

可直接使用的图在：

```text
outputs/C_decile_mean_target.svg
outputs/C_policy_spread.svg
outputs/C_top_factor_spread.svg
outputs/C_execution_limit_equity_curve.svg
outputs/C_AGENT_DASHBOARD.md
outputs/C_agent_decision_examples.md
outputs/C_AGENT_GUI.html
```
