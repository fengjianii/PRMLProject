# C: Lightweight Trading Agent

本模块是 C 同学负责的轻量 Trading Agent 扩展。它不替代课程主任务，也不改变
`python meow.py` 的默认训练和评价流程。

当前主线已经同步 A/B 最新进展：

- A: `feat.py` 已扩展为 76 个特征，包含 `cs_rank`、P0 rolling stats 和 P1 cross-sectional features。
- B: `mdl.py` 默认使用调参后的 `HistGradientBoostingRegressor` 参数。
- C: 在预测模型之后接入轻量 Agent 决策层，用来验证 `forecast` 是否有交易方向价值。

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

## 汇报边界

推荐说法：

```text
我们没有把项目包装成完整 Agent trading 系统，因为课程任务本身是 12 分钟收益率预测。
我们做的是 Agent-assisted alpha mining + lightweight trading decision prototype。
预测模型负责从 76 个市场状态特征中输出 forecast，Agent 层再把 forecast 转成
buy / hold / sell 动作，并用真实 fret12 做简化 reward 验证。
```
