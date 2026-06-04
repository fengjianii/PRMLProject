# C: Lightweight Trading Agent

本模块是 C 同学负责的轻量 Trading Agent 扩展。它不替代课程主任务，也不改变
`python meow.py` 的默认训练和评价流程。

当前主线已经同步 A/B 最新进展：

- A: `feat.py` 已扩展为 76 个特征，包含 `cs_rank`、P0 rolling stats 和 P1 cross-sectional features。
- B: `GBT-final` 使用调参后的 LightGBM GBT 模型，并保留 feature set 开关。
- C: 在预测模型之后接入轻量 Agent 决策层，用来验证 `forecast` 是否有交易方向价值。
- B 的最新 `GBT-final` 调参报告显示最佳 Pearson 为 0.0655；真实数据完整复现后，测试集 Pearson 为 0.0677。

## 定位

课程评分主线仍然是：

```text
市场数据 -> 特征工程 -> 模型预测 fret12 -> MSE / Pearson / R2
```

C 模块只在预测之后加一层：

```text
forecast -> buy / hold / sell -> simplified reward
```

这不是完整自动交易系统，不模拟撮合、排队、滑点、撤单或真实资金曲线。它只是一个
signal backtest，用来说明预测信号能否转成可解释的交易动作。

## 文件

- `agent_policy.py`: 把 `forecast` 转成 `action`。
- `backtest.py`: 计算 `action * fret12`，并扣除可选交易成本。
- `run_c_experiment.py`: 基于 B 的预测结果生成策略对比和 forecast decile 分析。
- `agent_factor_mining.py`: 自动生成特征、挖掘因子 IC / 分组收益差，并导出 factor-aware Agent 配置。
- `agent_execution_sim.py`: 简化执行仿真，加入撮合、滑点、订单排队、仓位上限和资金曲线。
- `agent_rl_policy.py`: 轻量 Q-learning / contextual-bandit 策略选择器，用历史 reward 在线更新策略偏好。
- `agent_controller.py`: 汇总预测回测、因子挖掘、执行仿真和 RL 输出，生成最终 Dashboard 和交易决策样例。
- `agent_gui.py`: 生成深色静态 GUI，把 C 的指标、图表、执行结果和交易样例做成可直接演示的页面。
- `meow.py`: 保留默认入口，只通过环境变量开启预测导出和 Agent 摘要。
- `C_AGENT_REPORT.md`: 可直接放进报告的 C 部分中文材料。

## 默认运行

默认命令不变：

```bash
python meow.py
```

默认行为仍然是训练 2023-06-01 到 2023-11-30，评价 2023-12-01 到
2023-12-29。

## 导出预测结果

PowerShell:

```powershell
$env:MEOW_PREDICTION_OUTPUT = "outputs/prediction_output.csv"
python meow.py
```

导出的 CSV 至少包含：

```text
symbol,date,interval,fret12,forecast
```

如果使用 B 的最终模型目录，请在 `GBT-final/` 下运行：

```powershell
cd GBT-final
$env:MEOW_PREDICTION_OUTPUT = "../outputs/B_prediction_output.csv"
python meow.py
```

`GBT-final/meow.py` 默认读取项目根目录的 `data/`，默认加载
`GBT-final/tuning_output/best_params.json`。如果数据放在其他目录，可以用
`MEOW_DATA_DIR` 覆盖。

然后回到项目根目录，用同一个 `backtest.py` 跑 C 的 Agent：

```powershell
cd ..
python backtest.py --predictions outputs/B_prediction_output.csv --policy top_bottom --top-frac 0.10 --bottom-frac 0.10 --cost-bps 1 --summary-output outputs/B_agent_summary.csv
```

如果要生成 C 的策略对比和 decile 分析，继续运行：

```powershell
python run_c_experiment.py --predictions outputs/B_prediction_output.csv --output-dir outputs --cost-bps 1
```

## 训练后直接跑 Agent 摘要

PowerShell:

```powershell
$env:MEOW_PREDICTION_OUTPUT = "outputs/prediction_output.csv"
$env:MEOW_AGENT_SUMMARY_OUTPUT = "outputs/agent_summary.txt"
python meow.py
```

`meow.py` 内置的 Agent 摘要使用默认设置：

```text
policy = top_bottom
top_frac = 0.10
bottom_frac = 0.10
cost_bps = 1
```

含义是：每个 `date + interval` 横截面里，做多预测最高的 10%，做空预测最低的
10%，并扣除 1 bps 简化交易成本。

## 单独运行 Agent 回测

如果 B 已经导出了预测文件，C 可以单独跑：

```bash
python backtest.py \
  --predictions outputs/prediction_output.csv \
  --policy top_bottom \
  --top-frac 0.10 \
  --bottom-frac 0.10 \
  --cost-bps 1 \
  --summary-output outputs/agent_summary.csv
```

长-only 版本：

```bash
python backtest.py \
  --predictions outputs/prediction_output.csv \
  --policy top_bottom \
  --top-frac 0.10 \
  --bottom-frac 0 \
  --long-only \
  --cost-bps 1
```

阈值版本：

```bash
python backtest.py \
  --predictions outputs/prediction_output.csv \
  --policy threshold \
  --buy-threshold 0.0005 \
  --sell-threshold -0.0005 \
  --cost-bps 1
```

## Agent 辅助因子挖掘口径

报告里不要说 Agent 直接决定交易。更稳的说法是：

```text
Agent 作为研究助手，根据市场微观结构、字段含义和已有实验结果，把候选因子组织成若干
可解释的因子族。最终是否采用这些因子，仍然由验证集 MSE、Pearson 和 R2 决定。
```

当前 76 个特征可以归成这些因子族：

1. 盘口压力: `cs_rank_ob_imb0`, `cs_rank_ob_imb4`, `cs_rank_ob_imb9`,
   `cs_rank_buy_pressure_0`, `cs_rank_buy_pressure_4`, `buy_pressure_9`,
   `cs_rank_amount_imb_4`, `amount_imb_9`, `amount_imb_19`。
2. 流动性和价差: `cs_rank_spread`, `cs_rank_relative_spread`,
   `cs_rank_weighted_spread_4`, `cs_rank_depth_sum_4`, `depth_sum_9`,
   `cs_rank_depth_delta_4`, `depth_delta_9`。
3. 成交主动性: `cs_rank_trade_buy_intensity`, `trade_sell_intensity`,
   `cs_rank_trade_net_intensity`, turnover-ratio features,
   `cs_rank_trade_count_imb`, `trade_count_imbema5`。
4. 动量和反转: `ret_1`, `ret_3`, `ret_6`, `ret_12`, `ret_24`,
   `rolling_mean_ret_6`, `rolling_mean_ret_12`, `rolling_vol_6`,
   `rolling_vol_12`。
5. P0 时序状态: `ob_imb0_roll_mean_12`, `ob_imb0_roll_std_12`,
   `trade_imb_roll_mean_12`, `trade_imb_roll_std_12`,
   `buy_intensity_roll_mean_12`, `*_change_6`。
6. P1 横截面相对强弱: `cs_rank_ret_3`, `cs_rank_ret_12`,
   `cs_rank_rolling_mean_ret_12`, `cs_rank_rolling_vol_12`,
   `cs_rank_high_low_range`, `cs_rank_ret_3_change`。
7. 交互因子: `ret_3_x_imb0`, `ret_6_x_imb0`,
   `ret_3_x_buy_intensity`, `spread_x_vol`, `highlow_x_imb0`。

B 最新调参结果显示，`GBT-final/tuning_output/best_params.json` 中的最佳模型
Pearson 为 0.0655。真实数据复现后，B 最终模型在 2023 年 12 月测试集上
Pearson 为 0.0677，当前 C 的 Agent 层应该优先使用这个最终模型导出的
`forecast`，而不是早期根目录模型输出。

## 真实数据结果

数据集为 Kaggle `MEOW dataset` version 3，共 144 个 `.h5` 文件。训练区间为
2023-06-01 到 2023-11-30，测试区间为 2023-12-01 到 2023-12-29。

根目录 A 模型真实数据结果：

```text
Pearson = 0.0511
R2 = 0.00196
top-bottom 10% long_short_spread = 0.00069596
```

B 的 `GBT-final` 模型真实数据结果：

```text
Pearson = 0.0677
R2 = 0.00443
best_iteration = 935
```

C 使用 B 预测结果的默认 Agent 回测结果：

```text
policy = top_bottom
top_frac = 0.10
bottom_frac = 0.10
cost_bps = 1
samples = 1,462,884
trade_ratio = 0.1986
mean_net_return = 0.00005713
trade_mean_net_return = 0.00028772
hit_rate = 0.5008
long_mean_target = 0.00043309
short_mean_target = -0.00034123
long_short_spread = 0.00077432
```

Forecast decile 分析中，底部十分位平均真实收益为 -3.41 bps，顶部十分位平均真实收益为
4.33 bps，顶部减底部约为 7.74 bps。这是 C 最好讲的一组结果：模型预测值不只提高
Pearson，也能被转成有方向的 buy / hold / sell 信号。

按 `date + interval` 做横截面期级聚合后，测试集共有 4,746 个横截面期。默认
top-bottom 10% 策略的期级平均多空差为 0.00077266，约 7.73 bps；其中
72.19% 的横截面期多空差为正。这是明天汇报时更稳的防守口径：C 不只看逐行样本均值，
也检查了每个横截面期的排序方向。

`run_c_experiment.py` 会生成以下 C 汇报产物：

```text
outputs/C_policy_comparison.csv
outputs/C_decile_analysis.csv
outputs/C_default_period_returns.csv
outputs/C_experiment_summary.md
outputs/C_decile_mean_target.svg
outputs/C_policy_spread.svg
```

## 自动因子挖掘 Agent

C 现在不只做预测结果回测，还支持自动因子挖掘。运行：

```powershell
python agent_factor_mining.py --data-dir data --start-date 20231201 --end-date 20231229 --output-dir outputs --top-k 15 --factor-agent-top-n 8 --cost-bps 1
```

它会自动完成：

1. 从 h5 数据生成 76 个特征。
2. 对每个因子计算全局 Pearson / Spearman。
3. 对每个 `date + interval` 横截面计算 period IC、ICIR 和 IC 正负比例。
4. 对每个因子做 top-bottom 10% 分组收益差。
5. 按因子家族聚合，输出 Agent 建议。
6. 根据 Top 因子自动生成 `factor-aware Agent` 配置，并做诊断性回测。

输出文件：

```text
outputs/C_factor_mining_summary.csv
outputs/C_factor_family_summary.csv
outputs/C_factor_mining_report.md
outputs/C_top_factor_spread.svg
outputs/C_factor_agent_config.json
outputs/C_factor_agent_summary.csv
```

12 月真实数据诊断中，Agent 自动挖出的最强单因子方向主要是短周期反转：

```text
cs_rank_ret_3: mean_ic = -0.049636, spread = -4.45 bps
ret_1: mean_ic = -0.060702, spread = -5.63 bps
ret_3: mean_ic = -0.060214, spread = -4.45 bps
ret_3_x_buy_intensity: mean_ic = -0.057776, spread = -4.09 bps
```

这组结果的解释是：短时间内横截面涨幅越靠前，未来 12 分钟越容易回撤，所以这些因子应该作为
reverse / short / risk signal 使用。因子家族层面，`P1_cross_section`、`momentum_reversal`
和 `interaction` 是当前最值得优先解释和做消融的方向。

自动生成的 factor-aware Agent 使用 Top 8 个挖掘因子组合成 `agent_factor_score`，诊断结果为：

```text
long_short_spread = 0.00048814，约 4.88 bps
period_long_short_spread = 0.00048702，约 4.87 bps
period_positive_spread_rate = 0.6764
```

这个结果不替代 B 的最终 forecast，也不当作最终 out-of-sample 证明。它的价值是让 Agent 能自动提出候选因子、解释因子方向，并生成下一轮可验证的策略配置。

## 简化执行仿真层

为了让 Trading Agent 更接近真实交易链路，C 新增了 `agent_execution_sim.py`。它不是完整交易所撮合引擎，但会把原来的 signal backtest 往执行层推进一步：

```text
forecast -> action -> order -> matching / queue / slippage -> position PnL -> equity curve
```

运行示例：

```powershell
python agent_execution_sim.py --predictions outputs/B_prediction_output.csv --data-dir data --output-dir outputs --output-prefix C_execution_limit --order-style limit --notional-per-trade 10000 --max-gross-notional 10000000 --queue-ahead-frac 0.50 --impact-k 2 --fee-bps 0.5 --exit-impact-bps 0.5
```

当前实现包含：

1. 撮合近似：market order 直接穿越买卖价差；limit order 只有当对手方成交量超过估计排队量时成交。
2. 滑点建模：市价单根据下单量与盘口五档深度的参与率加入冲击成本。
3. 订单排队：用 `queue_ahead_frac * bsize0/asize0` 估计前方排队量，未成交部分视为撤单。
4. 仓位管理：每笔订单有 `notional_per_trade`，每个 `date + interval` 有 `max_gross_notional` 总敞口上限。
5. 资金曲线：把每个横截面期的 PnL 聚合成 equity curve，并计算 drawdown。

三种执行模式的真实数据诊断结果如下：

```text
market:
  fill_rate = 1.0000
  final_equity = 8,331,537.84
  total_pnl = -1,668,462.16
  max_drawdown = -0.1672

hybrid:
  fill_rate = 1.0000
  final_equity = 9,778,965.38
  total_pnl = -221,034.62
  max_drawdown = -0.0312

limit:
  fill_rate = 0.6628
  mean_fill_ratio = 0.6513
  final_equity = 10,669,876.40
  total_pnl = 669,876.40
  max_drawdown = -0.0020
```

这个结果很适合汇报：signal 层面有 alpha，但市价强成交会被价差和滑点吃掉；被动限价挂单虽然只有约 66% 的订单成交，但执行成本更低，资金曲线反而更稳。这说明交易动作不能只看预测方向，还必须考虑订单类型和执行方式。

输出文件包括：

```text
outputs/C_execution_market_summary.csv
outputs/C_execution_hybrid_summary.csv
outputs/C_execution_limit_summary.csv
outputs/C_execution_limit_equity_curve.svg
outputs/C_execution_limit_report.md
```

## 轻量强化学习策略选择

C 还新增了 `agent_rl_policy.py`，作为轻量强化学习 / contextual-bandit 原型。它不做深度强化学习，而是在每个 `date + interval` 上根据状态选择策略：

```text
state = forecast dispersion + realized volatility bucket
action = hold / top-bottom 5% / top-bottom 10% / top-bottom 20%
reward = cost-adjusted mean net return
update = Q[state, action] <- Q + alpha * (reward - Q)
```

运行：

```powershell
python agent_rl_policy.py --predictions outputs/B_prediction_output.csv --output-dir outputs --cost-bps 1 --alpha 0.20 --epsilon 0.05
```

真实数据诊断结果：

```text
steps = 4,746
final_equity = 10,268,736.34
total_pnl = 268,736.34
mean_reward = 0.00031033
positive_reward_rate = 0.7097
max_drawdown = -0.00038693
most_used_action = hold
```

这里的意义不是证明强化学习策略已经最优，而是补上“得到 reward 后更新策略偏好”的闭环。它学到在高风险或弱信号状态下更多选择 hold，在部分 strong-signal 状态下选择 top-bottom 20% 或 5%。

## Agent Controller 总控层

最后新增 `agent_controller.py`，用于把 C 分散的实验结果收束成一个汇报入口。它不重新训练模型，只读取已有输出，生成最终 Dashboard 和可解释交易样例：

```powershell
python agent_controller.py --predictions outputs/B_prediction_output.csv --output-dir outputs
```

它会读取：

```text
outputs/C_policy_comparison.csv
outputs/C_decile_analysis.csv
outputs/C_default_period_returns.csv
outputs/C_factor_mining_summary.csv
outputs/C_factor_agent_summary.csv
outputs/C_execution_market_summary.csv
outputs/C_execution_hybrid_summary.csv
outputs/C_execution_limit_summary.csv
outputs/C_rl_policy_summary.csv
outputs/C_rl_q_table.csv
```

并输出：

```text
outputs/C_AGENT_DASHBOARD.md
outputs/C_agent_decision_examples.csv
outputs/C_agent_decision_examples.md
```

Dashboard 会自动汇总明天最该讲的数字：B 最终模型 Pearson `0.0677`，C 默认信号多空差 `7.74 bps`，期级正多空差比例 `72.19%`，自动因子挖掘的代表因子 `cs_rank_ret_3`，limit 执行仿真的最终资金 `10,669,876.40`，以及 RL selector 的最终资金 `10,268,736.34`。

交易样例会从 B 的预测文件中各抽几条 buy / sell / hold 记录，展示 Agent 如何根据同一横截面的 forecast percentile 做动作选择。这个文件很适合应对老师追问：“你说 Agent 会判断，那具体某一条样本是怎么判断的？”

## 静态 GUI

如果要汇报展示，可以继续运行：

```powershell
python agent_gui.py --output-dir outputs
```

页面输出：

```text
outputs/C_AGENT_GUI.html
```

`C_AGENT_GUI.html` 是一个无需前端框架的静态深色 dashboard，风格参考 Umami 的左侧导航和指标工作台。它会读取 C 的已有输出，展示：

1. 核心 KPI：7.74 bps 多空差、72.19% 期级正多空差比例、10.67M limit equity、10.27M RL equity。
2. Signal 图：forecast decile 收益和 top-bottom 策略敏感性。
3. Execution 图：limit 资金曲线和 market / hybrid / limit 执行对比。
4. Factor mining 和 RL selector：Top 因子、Q table 偏好。
5. Decision tape：具体 buy / sell / hold 样例。

这个 GUI 只用于展示和研究复盘，不改变实验结果，也不作为生产交易界面。

## 汇报边界

推荐说法：

```text
我们没有把项目包装成完整 Agent trading 系统，因为课程任务本身是 12 分钟收益率预测。
我们做的是 Agent-assisted alpha mining + lightweight trading decision prototype。
预测模型负责从 76 个市场状态特征中输出 forecast，Agent 层再把 forecast 转成
buy / hold / sell 动作，并用真实 fret12 做简化 reward 验证。
```
