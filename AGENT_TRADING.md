# Lightweight Trading Agent Notes

Owner: C

This module is an add-on to the course prediction task. The main project still
optimizes `fret12` prediction with MSE, Pearson correlation, and R2. The Agent
layer only checks whether the model forecast can be converted into a simple
buy / hold / sell signal.

## Files

- `agent_policy.py`: converts `forecast` into `action`.
- `backtest.py`: evaluates `action * fret12` with an optional transaction cost.
- `meow.py`: keeps the original training/evaluation flow, with optional outputs
  controlled by environment variables.

## Default Behavior

The default command is unchanged:

```bash
python meow.py
```

It trains on June-November 2023 and evaluates on December 2023.

## Export Predictions

Set `MEOW_PREDICTION_OUTPUT` to save a CSV with:

```text
symbol,date,interval,fret12,forecast
```

Example on PowerShell:

```powershell
$env:MEOW_PREDICTION_OUTPUT = "outputs/prediction_output.csv"
python meow.py
```

## Run Agent Backtest During Evaluation

Set `MEOW_AGENT_SUMMARY_OUTPUT` to run the default top-bottom Agent backtest
after model evaluation. This uses a 1 bps cost by default inside `meow.py`.

```powershell
$env:MEOW_AGENT_SUMMARY_OUTPUT = "outputs/agent_summary.txt"
python meow.py
```

Both outputs can be enabled together:

```powershell
$env:MEOW_PREDICTION_OUTPUT = "outputs/prediction_output.csv"
$env:MEOW_AGENT_SUMMARY_OUTPUT = "outputs/agent_summary.txt"
python meow.py
```

## Standalone Backtest

If B exports predictions, C can run the Agent layer separately:

```bash
python backtest.py \
  --predictions outputs/prediction_output.csv \
  --policy top_bottom \
  --top-frac 0.10 \
  --bottom-frac 0.10 \
  --cost-bps 1 \
  --summary-output outputs/agent_summary.csv
```

For a long-only version:

```bash
python backtest.py \
  --predictions outputs/prediction_output.csv \
  --policy top_bottom \
  --top-frac 0.10 \
  --bottom-frac 0 \
  --long-only \
  --cost-bps 1
```

## Report Wording

Use this positioning in the report:

```text
The project is not a full automatic trading system. The course target remains
12-minute forward-return prediction. We add a lightweight Trading Agent after
the predictor: the Agent receives `forecast`, converts it into buy / hold / sell
actions with a cross-sectional top-bottom policy, and uses realized `fret12` as
a simplified reward. The module does not simulate exchange matching, queue
position, slippage, or order cancellation. It only validates whether the alpha
signal has directional trading value.
```

## Agent-Assisted Factor Mining

The Agent-assisted feature discussion should map existing `feat.py` features
into five factor families:

1. Order-book pressure: `ob_imb0`, `ob_imb4`, `ob_imb9`,
   `buy_pressure_0`, `buy_pressure_4`, `buy_pressure_9`.
2. Liquidity and spread: `spread`, `relative_spread`, `weighted_spread_4`.
3. Trade aggressiveness: `trade_buy_intensity`, `trade_sell_intensity`,
   `trade_net_intensity`, turnover-ratio features, and trade-count imbalance.
4. Momentum / reversal: `ret_1`, `ret_3`, `ret_6`, `ret_12`, `ret_24`,
   rolling mean, and rolling volatility features.
5. Interactions: `ret_3_x_imb0`, `ret_6_x_imb0`,
   `ret_3_x_buy_intensity`, `spread_x_vol`, `highlow_x_imb0`.

Final feature adoption should still be justified by validation metrics, not by
the Agent suggestion alone.
