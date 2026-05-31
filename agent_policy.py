import numpy as np
import pandas as pd


ACTION_COL = "action"


def as_prediction_frame(df):
    """Return a flat prediction frame with symbol/date/interval columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if isinstance(df.index, pd.MultiIndex):
        return df.reset_index()
    return df.copy()


def _require_columns(df, columns):
    missing = [x for x in columns if x not in df.columns]
    if missing:
        raise ValueError("Missing required columns: {}".format(", ".join(missing)))


def top_bottom_policy(
        df,
        prediction_col="forecast",
        group_cols=("date", "interval"),
        top_frac=0.10,
        bottom_frac=0.10,
        allow_short=True):
    """Assign actions by cross-sectional forecast ranks.

    Actions:
      1: long the top forecast bucket
      0: hold / no trade
     -1: short the bottom forecast bucket when allow_short is True
    """
    if not 0 < top_frac <= 1:
        raise ValueError("top_frac must be in (0, 1]")
    if not 0 <= bottom_frac <= 1:
        raise ValueError("bottom_frac must be in [0, 1]")
    if top_frac + bottom_frac > 1:
        raise ValueError("top_frac + bottom_frac must not exceed 1")

    out = as_prediction_frame(df)
    _require_columns(out, [prediction_col] + list(group_cols))
    out[ACTION_COL] = 0

    ranks = out.groupby(list(group_cols))[prediction_col].rank(
        method="first", pct=True)
    out.loc[ranks > 1 - top_frac, ACTION_COL] = 1
    if allow_short and bottom_frac > 0:
        out.loc[ranks <= bottom_frac, ACTION_COL] = -1
    return out


def threshold_policy(
        df,
        prediction_col="forecast",
        buy_threshold=0.0,
        sell_threshold=0.0,
        allow_short=True):
    """Assign actions with fixed forecast thresholds."""
    out = as_prediction_frame(df)
    _require_columns(out, [prediction_col])
    out[ACTION_COL] = 0
    out.loc[out[prediction_col] > buy_threshold, ACTION_COL] = 1
    if allow_short:
        out.loc[out[prediction_col] < sell_threshold, ACTION_COL] = -1
    return out


def describe_actions(df, action_col=ACTION_COL):
    """Summarize action counts for logs and reports."""
    out = as_prediction_frame(df)
    _require_columns(out, [action_col])
    counts = out[action_col].value_counts().to_dict()
    total = float(len(out)) if len(out) else 1.0
    return {
        "samples": int(len(out)),
        "long_count": int(counts.get(1, 0)),
        "short_count": int(counts.get(-1, 0)),
        "hold_count": int(counts.get(0, 0)),
        "trade_ratio": float(
            (counts.get(1, 0) + counts.get(-1, 0)) / total),
    }


def clean_prediction_values(df, columns=("forecast", "fret12")):
    """Replace NaN/Inf in prediction and target columns."""
    out = as_prediction_frame(df)
    present = [x for x in columns if x in out.columns]
    out[present] = out[present].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
