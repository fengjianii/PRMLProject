import argparse
import html
import os
from datetime import datetime

import numpy as np
import pandas as pd

from agent_controller import (
    best_factor_row,
    first_row,
    fmt_bps,
    fmt_decimal,
    fmt_money,
    fmt_number,
    fmt_pct,
    read_outputs,
    select_row,
    value,
)


CSS = r"""
:root {
  color-scheme: dark;
  --bg: #0b0d10;
  --shell: #101216;
  --panel: #15181d;
  --panel-2: #111419;
  --line: #272c34;
  --line-strong: #343a45;
  --text: #f3f6fb;
  --muted: #8d96a5;
  --soft: #c9d1dc;
  --blue: #3b82f6;
  --green: #22c55e;
  --red: #f05252;
  --amber: #f4b740;
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  letter-spacing: 0;
}

a {
  color: inherit;
  text-decoration: none;
}

.app {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 248px minmax(0, 1fr);
  background:
    linear-gradient(90deg, rgba(255, 255, 255, 0.035) 0, transparent 1px) 248px 0 / 1px 100% no-repeat,
    var(--bg);
}

.sidebar {
  position: sticky;
  top: 0;
  height: 100vh;
  padding: 18px 14px;
  background: #0f1115;
  border-right: 1px solid var(--line);
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  height: 34px;
  margin-bottom: 24px;
  color: var(--text);
  font-size: 13px;
  font-weight: 700;
}

.brand-mark {
  width: 20px;
  height: 20px;
  display: inline-grid;
  place-items: center;
  border-radius: 5px;
  background: #f3f6fb;
  color: #0b0d10;
  font-size: 11px;
  font-weight: 800;
}

.back {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--soft);
  font-size: 12px;
  margin-bottom: 26px;
}

.nav-title {
  color: var(--soft);
  font-size: 11px;
  font-weight: 700;
  margin: 18px 0 8px;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 34px;
  padding: 0 10px;
  border-radius: 6px;
  color: var(--soft);
  font-size: 12px;
  font-weight: 600;
}

.nav-link:hover,
.nav-link.active {
  background: #1c2027;
  color: var(--text);
}

.nav-dot {
  width: 6px;
  height: 6px;
  border-radius: 99px;
  background: var(--muted);
}

.nav-link.active .nav-dot {
  background: var(--blue);
  box-shadow: 0 0 18px rgba(59, 130, 246, 0.85);
}

.sidebar-foot {
  position: absolute;
  left: 14px;
  right: 14px;
  bottom: 16px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 11px;
  line-height: 1.7;
}

.main {
  min-width: 0;
  padding: 18px 30px 56px;
}

.topbar {
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: var(--soft);
  font-size: 12px;
  border-bottom: 1px solid var(--line);
}

.topbar-left,
.topbar-right {
  display: flex;
  align-items: center;
  gap: 18px;
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 28px;
  padding: 0 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #101318;
  color: var(--soft);
  font-size: 12px;
  font-weight: 600;
}

.content {
  width: min(1280px, 100%);
  margin: 0 auto;
}

.hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 20px;
  align-items: end;
  padding: 28px 0 18px;
  border-bottom: 1px solid var(--line);
}

.eyebrow {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}

h1 {
  margin: 6px 0 0;
  font-size: clamp(32px, 4vw, 52px);
  line-height: 0.98;
  letter-spacing: 0;
}

.hero-meta {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.section {
  padding: 26px 0 0;
}

.section-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 12px;
}

.section-title {
  margin: 0;
  color: var(--text);
  font-size: 15px;
  font-weight: 800;
}

.section-note {
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
}

.metric {
  min-height: 114px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
}

.metric:hover {
  transform: translateY(-2px);
  border-color: var(--line-strong);
  background: #181c22;
}

.metric-label {
  color: var(--soft);
  font-size: 11px;
  font-weight: 800;
}

.metric-value {
  margin-top: 14px;
  color: var(--text);
  font-size: clamp(24px, 3vw, 34px);
  line-height: 1;
  font-weight: 850;
}

.metric-sub {
  display: inline-flex;
  align-items: center;
  margin-top: 12px;
  padding: 4px 7px;
  border-radius: 4px;
  background: rgba(34, 197, 94, 0.14);
  color: var(--green);
  font-size: 11px;
  font-weight: 800;
}

.metric-sub.red {
  background: rgba(240, 82, 82, 0.14);
  color: var(--red);
}

.metric-sub.blue {
  background: rgba(59, 130, 246, 0.14);
  color: #7cb2ff;
}

.grid-2 {
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(340px, 0.75fr);
  gap: 12px;
}

.grid-3 {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel-2);
  overflow: hidden;
}

.panel-head {
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 14px;
  border-bottom: 1px solid var(--line);
}

.panel-title {
  font-size: 12px;
  font-weight: 800;
  color: var(--text);
}

.panel-kicker {
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
}

.panel-body {
  padding: 14px;
}

.chart svg {
  display: block;
  width: 100%;
  height: auto;
}

.bars {
  display: grid;
  gap: 12px;
}

.bar-row {
  display: grid;
  grid-template-columns: 68px minmax(0, 1fr) 98px;
  gap: 10px;
  align-items: center;
  color: var(--soft);
  font-size: 12px;
  font-weight: 700;
}

.bar-track {
  height: 8px;
  border-radius: 4px;
  background: #20252d;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 4px;
  background: var(--blue);
  transform-origin: left;
  animation: growX 700ms ease both;
}

.bar-fill.good {
  background: var(--green);
}

.bar-fill.bad {
  background: var(--red);
}

.factor-list,
.q-list {
  display: grid;
  gap: 8px;
}

.list-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 88px 72px;
  gap: 8px;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid var(--line);
}

.list-row:last-child {
  border-bottom: 0;
}

.name {
  min-width: 0;
  color: var(--text);
  font-size: 12px;
  font-weight: 800;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.muted {
  color: var(--muted);
  font-size: 11px;
  font-weight: 650;
}

.value-green {
  color: var(--green);
}

.value-red {
  color: var(--red);
}

.value-blue {
  color: #7cb2ff;
}

.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 880px;
}

th,
td {
  padding: 12px 10px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  font-size: 12px;
}

th {
  color: var(--muted);
  font-size: 11px;
  font-weight: 800;
}

td {
  color: var(--soft);
  font-weight: 650;
}

.decision {
  display: inline-flex;
  min-width: 76px;
  justify-content: center;
  padding: 5px 8px;
  border-radius: 5px;
  color: var(--text);
  font-size: 11px;
  font-weight: 850;
}

.decision.buy {
  background: rgba(34, 197, 94, 0.15);
  color: var(--green);
}

.decision.sell_short {
  background: rgba(240, 82, 82, 0.15);
  color: var(--red);
}

.decision.hold {
  background: rgba(141, 150, 165, 0.16);
  color: var(--soft);
}

.boundary {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.statement {
  padding: 14px;
  min-height: 94px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #111419;
}

.statement strong {
  display: block;
  margin-bottom: 8px;
  color: var(--text);
  font-size: 12px;
}

.statement span {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.65;
}

.line-chart path.main-line {
  stroke-dasharray: 1200;
  stroke-dashoffset: 1200;
  animation: drawLine 1200ms ease forwards;
}

@keyframes fadeUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes growX {
  from {
    transform: scaleX(0);
  }
  to {
    transform: scaleX(1);
  }
}

@keyframes drawLine {
  to {
    stroke-dashoffset: 0;
  }
}

@media (max-width: 1040px) {
  .app {
    grid-template-columns: 1fr;
  }

  .sidebar {
    position: static;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }

  .sidebar-foot {
    position: static;
    margin-top: 18px;
  }

  .metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .grid-2,
  .grid-3,
  .boundary {
    grid-template-columns: 1fr;
  }

  .hero {
    grid-template-columns: 1fr;
  }

  .hero-meta {
    justify-content: flex-start;
  }
}

@media (max-width: 640px) {
  .main {
    padding: 14px 16px 40px;
  }

  .topbar {
    align-items: flex-start;
    height: auto;
    gap: 12px;
    padding-bottom: 14px;
    flex-direction: column;
  }

  .metrics {
    grid-template-columns: 1fr;
  }

  .section-head {
    align-items: flex-start;
    flex-direction: column;
    gap: 6px;
  }

  .section-note {
    max-width: 100%;
  }
}
"""


SCRIPT = r"""
const links = Array.from(document.querySelectorAll(".nav-link"));
const sections = links
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

const activate = () => {
  const current = sections
    .slice()
    .reverse()
    .find((section) => section.getBoundingClientRect().top < 180);
  if (!current) return;
  links.forEach((link) => {
    link.classList.toggle("active", link.getAttribute("href") === `#${current.id}`);
  });
};

document.addEventListener("scroll", activate, { passive: true });
activate();
"""


def esc(value):
    if pd.isna(value):
        return ""
    return html.escape(str(value), quote=True)


def safe_float(value_to_cast, default=0.0):
    try:
        if pd.isna(value_to_cast):
            return default
        return float(value_to_cast)
    except (TypeError, ValueError):
        return default


def fmt_money_compact(value_to_format):
    value_float = safe_float(value_to_format, default=np.nan)
    if pd.isna(value_float):
        return "missing"
    abs_value = abs(value_float)
    if abs_value >= 1000000:
        return "{:.2f}M".format(value_float / 1000000.0)
    if abs_value >= 1000:
        return "{:.1f}K".format(value_float / 1000.0)
    return "{:.2f}".format(value_float)


def clamp(value_to_clamp, lower, upper):
    return max(lower, min(upper, value_to_clamp))


def read_csv_if_exists(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_examples(output_dir):
    return read_csv_if_exists(os.path.join(output_dir, "C_agent_decision_examples.csv"))


def load_equity_curve(output_dir):
    return read_csv_if_exists(os.path.join(output_dir, "C_execution_limit_equity_curve.csv"))


def svg_bar_chart(labels, values, width=820, height=280):
    values = [safe_float(v) for v in values]
    if not values:
        return empty_chart(width, height, "No data")

    left, right, top, bottom = 52, 18, 18, 46
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_v = min(0.0, min(values))
    max_v = max(0.0, max(values))
    if min_v == max_v:
        min_v -= 1.0
        max_v += 1.0
    pad = (max_v - min_v) * 0.12
    min_v -= pad
    max_v += pad

    def y_pos(v):
        return top + (max_v - v) / (max_v - min_v) * plot_h

    zero_y = y_pos(0.0)
    bar_gap = 10
    bar_w = (plot_w - bar_gap * (len(values) - 1)) / len(values)
    parts = [
        '<svg viewBox="0 0 {0} {1}" role="img" aria-label="bar chart">'.format(width, height),
        '<rect width="100%" height="100%" fill="transparent"/>',
    ]

    for i in range(5):
        y = top + plot_h * i / 4
        parts.append(
            '<line x1="{0}" y1="{1:.2f}" x2="{2}" y2="{1:.2f}" stroke="#252b34" stroke-width="1"/>'.format(
                left, y, width - right))

    parts.append(
        '<line x1="{0}" y1="{1:.2f}" x2="{2}" y2="{1:.2f}" stroke="#394150" stroke-width="1"/>'.format(
            left, zero_y, width - right))

    for i, (label, val) in enumerate(zip(labels, values)):
        x = left + i * (bar_w + bar_gap)
        y = min(y_pos(val), zero_y)
        h = abs(y_pos(val) - zero_y)
        color = "#22c55e" if val >= 0 else "#f05252"
        parts.append(
            '<rect x="{0:.2f}" y="{1:.2f}" width="{2:.2f}" height="{3:.2f}" rx="3" fill="{4}" opacity="0.92"/>'.format(
                x, y, bar_w, max(h, 1.5), color))
        parts.append(
            '<text x="{0:.2f}" y="{1}" text-anchor="middle" fill="#8d96a5" font-size="10" font-weight="700">{2}</text>'.format(
                x + bar_w / 2, height - 18, esc(label)))
        label_y = y - 7 if val >= 0 else y + h + 13
        parts.append(
            '<text x="{0:.2f}" y="{1:.2f}" text-anchor="middle" fill="#c9d1dc" font-size="10" font-weight="800">{2:.2f}</text>'.format(
                x + bar_w / 2, label_y, val))

    parts.append("</svg>")
    return "\n".join(parts)


def svg_line_chart(values, width=820, height=280):
    values = [safe_float(v) for v in values if not pd.isna(v)]
    if not values:
        return empty_chart(width, height, "No equity curve")

    max_points = 180
    if len(values) > max_points:
        idx = np.linspace(0, len(values) - 1, max_points).astype(int)
        values = [values[i] for i in idx]

    left, right, top, bottom = 48, 18, 18, 36
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_v = min(values)
    max_v = max(values)
    if min_v == max_v:
        min_v -= 1.0
        max_v += 1.0
    pad = (max_v - min_v) * 0.08
    min_v -= pad
    max_v += pad

    def x_pos(i):
        if len(values) == 1:
            return left + plot_w
        return left + i / (len(values) - 1) * plot_w

    def y_pos(v):
        return top + (max_v - v) / (max_v - min_v) * plot_h

    points = ["{:.2f},{:.2f}".format(x_pos(i), y_pos(v)) for i, v in enumerate(values)]
    area = " ".join(points + [
        "{:.2f},{:.2f}".format(x_pos(len(values) - 1), top + plot_h),
        "{:.2f},{:.2f}".format(left, top + plot_h),
    ])

    parts = [
        '<svg class="line-chart" viewBox="0 0 {0} {1}" role="img" aria-label="equity curve">'.format(width, height),
        '<defs><linearGradient id="equityFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#3b82f6" stop-opacity="0.28"/><stop offset="100%" stop-color="#3b82f6" stop-opacity="0"/></linearGradient></defs>',
        '<rect width="100%" height="100%" fill="transparent"/>',
    ]
    for i in range(5):
        y = top + plot_h * i / 4
        parts.append(
            '<line x1="{0}" y1="{1:.2f}" x2="{2}" y2="{1:.2f}" stroke="#252b34" stroke-width="1"/>'.format(
                left, y, width - right))

    parts.append('<polygon points="{0}" fill="url(#equityFill)"/>'.format(area))
    parts.append(
        '<polyline class="main-line" points="{0}" fill="none" stroke="#3b82f6" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'.format(
            " ".join(points)))
    parts.append(
        '<text x="{0}" y="{1}" fill="#8d96a5" font-size="10" font-weight="700">start {2}</text>'.format(
            left, height - 12, fmt_money(values[0])))
    parts.append(
        '<text x="{0}" y="{1}" text-anchor="end" fill="#c9d1dc" font-size="10" font-weight="800">final {2}</text>'.format(
            width - right, height - 12, fmt_money(values[-1])))
    parts.append("</svg>")
    return "\n".join(parts)


def empty_chart(width, height, label):
    return (
        '<svg viewBox="0 0 {0} {1}"><rect width="100%" height="100%" fill="transparent"/>'
        '<text x="50%" y="50%" text-anchor="middle" fill="#8d96a5" font-size="12">{2}</text></svg>'
    ).format(width, height, esc(label))


def metric_card(label, main, sub, tone="green"):
    tone_class = "metric-sub {}".format(tone if tone != "green" else "")
    return """
<article class="metric">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{main}</div>
  <div class="{tone_class}">{sub}</div>
</article>
""".format(label=esc(label), main=esc(main), tone_class=tone_class.strip(), sub=esc(sub))


def execution_bars(rows):
    if not rows:
        return '<div class="muted">No execution data</div>'
    max_equity = max(safe_float(row.get("final_equity")) for row in rows)
    parts = ['<div class="bars">']
    for row in rows:
        style = row.get("order_style", "")
        equity = safe_float(row.get("final_equity"))
        fill_rate = safe_float(row.get("fill_rate"))
        width = 4 if max_equity <= 0 else clamp(equity / max_equity * 100, 4, 100)
        tone = "good" if equity >= 10000000 else "bad"
        parts.append("""
<div class="bar-row">
  <div>{style}</div>
  <div class="bar-track"><div class="bar-fill {tone}" style="width:{width:.2f}%"></div></div>
  <div>{equity}</div>
</div>
<div class="muted" style="margin-top:-8px;margin-left:78px;">fill {fill_rate}</div>
""".format(
            style=esc(style),
            tone=tone,
            width=width,
            equity=esc(fmt_money(equity)),
            fill_rate=esc(fmt_pct(fill_rate)),
        ))
    parts.append("</div>")
    return "\n".join(parts)


def factor_rows(frame, limit=7):
    if frame is None or len(frame) == 0:
        return '<div class="muted">No factor data</div>'
    parts = ['<div class="factor-list">']
    for _, row in frame.head(limit).iterrows():
        spread = safe_float(row.get("top_bottom_spread_bps"))
        tone = "value-green" if spread >= 0 else "value-red"
        parts.append("""
<div class="list-row">
  <div>
    <div class="name">{feature}</div>
    <div class="muted">{family}</div>
  </div>
  <div class="muted">IC {ic}</div>
  <div class="{tone}">{spread:.2f}</div>
</div>
""".format(
            feature=esc(row.get("feature", "")),
            family=esc(row.get("family", "")),
            ic=esc(fmt_decimal(row.get("mean_ic"), digits=4)),
            tone=tone,
            spread=spread,
        ))
    parts.append("</div>")
    return "\n".join(parts)


def q_rows(frame, limit=7):
    if frame is None or len(frame) == 0:
        return '<div class="muted">No RL data</div>'
    work = frame.copy()
    if "q_value" in work.columns:
        work = work.sort_values("q_value", ascending=False)
    parts = ['<div class="q-list">']
    for _, row in work.head(limit).iterrows():
        parts.append("""
<div class="list-row">
  <div>
    <div class="name">{state}</div>
    <div class="muted">{action}</div>
  </div>
  <div class="muted">Q</div>
  <div class="value-blue">{q}</div>
</div>
""".format(
            state=esc(row.get("state", "")),
            action=esc(row.get("action", "")),
            q=esc(fmt_decimal(row.get("q_value"), digits=5)),
        ))
    parts.append("</div>")
    return "\n".join(parts)


def decision_table(examples):
    if examples is None or len(examples) == 0:
        return '<div class="muted">No decision examples</div>'
    rows = []
    for _, row in examples.iterrows():
        decision = str(row.get("decision", "hold"))
        rows.append("""
<tr>
  <td>{symbol}</td>
  <td>{date}</td>
  <td>{interval}</td>
  <td>{pct}</td>
  <td>{decile}</td>
  <td><span class="decision {decision_class}">{decision}</span></td>
  <td>{expected}</td>
  <td>{realized}</td>
  <td>{reason}</td>
</tr>
""".format(
            symbol=esc(row.get("symbol", "")),
            date=esc(row.get("date", "")),
            interval=esc(row.get("interval", "")),
            pct=esc(fmt_pct(row.get("forecast_percentile"))),
            decile=esc(row.get("forecast_decile", "")),
            decision_class=esc(decision),
            decision=esc(decision.replace("_", " ")),
            expected=esc(fmt_number(row.get("expected_action_reward_bps"), digits=2)),
            realized=esc(fmt_number(row.get("realized_action_reward_bps"), digits=2)),
            reason=esc(row.get("reason", "")),
        ))
    return """
<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>Symbol</th>
        <th>Date</th>
        <th>Interval</th>
        <th>Forecast pct</th>
        <th>Decile</th>
        <th>Decision</th>
        <th>Expected bps</th>
        <th>Realized bps</th>
        <th>Reason</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>
""".format(rows="\n".join(rows))


def policy_rows(policy_frame):
    if policy_frame is None or len(policy_frame) == 0:
        return []
    rows = []
    for _, row in policy_frame.iterrows():
        rows.append({
            "label": str(row.get("policy_name", "")),
            "spread_bps": safe_float(row.get("long_short_spread")) * 10000.0,
        })
    return rows


def execution_rows(frames):
    rows = []
    for key in ["execution_market", "execution_hybrid", "execution_limit"]:
        row = first_row(frames.get(key))
        if row is not None:
            rows.append(row.to_dict())
    return rows


def build_html(output_dir, output_file):
    frames = read_outputs(output_dir)
    examples = load_examples(output_dir)
    equity = load_equity_curve(output_dir)

    policy = select_row(frames["policy"], "policy_name", "top_bottom_10_10")
    if policy is None:
        policy = first_row(frames["policy"])
    top5 = select_row(frames["policy"], "policy_name", "top_bottom_5_5")
    deciles = frames["decile"]
    factors = frames["factor"]
    factor = best_factor_row(factors)
    factor_agent = first_row(frames["factor_agent"])
    exec_limit = first_row(frames["execution_limit"])
    rl = first_row(frames["rl_summary"])

    decile_labels = []
    decile_values = []
    if deciles is not None and len(deciles):
        decile_labels = deciles["forecast_decile"].astype(str).tolist()
        if "mean_target_bps" in deciles.columns:
            decile_values = deciles["mean_target_bps"].tolist()
        else:
            decile_values = (deciles["mean_target"] * 10000.0).tolist()

    policies = policy_rows(frames["policy"])
    label_map = {
        "top_bottom_5_5": "5/5",
        "top_bottom_10_10": "10/10",
        "top_bottom_20_20": "20/20",
        "long_only_top_10": "long top 10",
    }
    policy_labels = [
        label_map.get(item["label"], item["label"].replace("top_bottom_", "").replace("_", "/"))
        for item in policies
    ]
    policy_values = [item["spread_bps"] for item in policies]

    equity_values = []
    if equity is not None and "equity" in equity.columns:
        equity_values = equity["equity"].tolist()

    exec_rows = execution_rows(frames)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    metrics = "\n".join([
        metric_card("Signal spread", fmt_bps(value(policy, "long_short_spread")), "top-bottom 10%"),
        metric_card("Positive periods", fmt_pct(value(policy, "period_positive_spread_rate")), "4,746 periods"),
        metric_card("Best policy", fmt_bps(value(top5, "long_short_spread")), "top-bottom 5/5", "blue"),
        metric_card("Limit equity", fmt_money_compact(value(exec_limit, "final_equity")), "fill {}".format(fmt_pct(value(exec_limit, "fill_rate")))),
        metric_card("RL equity", fmt_money_compact(value(rl, "final_equity")), "reward {}".format(fmt_pct(value(rl, "positive_reward_rate")))),
    ])

    body = """
<div class="app">
  <aside class="sidebar">
    <div class="brand"><span class="brand-mark">C</span><span>MEOW Agent</span></div>
    <a class="back" href="#overview">Back to overview</a>
    <div class="nav-title">Trading</div>
    <a class="nav-link active" href="#overview"><span class="nav-dot"></span>Overview</a>
    <a class="nav-link" href="#signal"><span class="nav-dot"></span>Signal</a>
    <a class="nav-link" href="#execution"><span class="nav-dot"></span>Execution</a>
    <a class="nav-link" href="#decisions"><span class="nav-dot"></span>Decisions</a>
    <div class="nav-title">Research</div>
    <a class="nav-link" href="#factor"><span class="nav-dot"></span>Factor mining</a>
    <a class="nav-link" href="#rl"><span class="nav-dot"></span>RL selector</a>
    <a class="nav-link" href="#boundary"><span class="nav-dot"></span>Boundary</a>
    <div class="sidebar-foot">
      <div>Branch: c-agent-backtest</div>
      <div>Generated: {generated_at}</div>
      <div>Source: B_prediction_output.csv</div>
    </div>
  </aside>

  <main class="main">
    <div class="content">
      <div class="topbar">
        <div class="topbar-left">
          <span>admin</span>
          <span>lucius / PRMLProject</span>
        </div>
        <div class="topbar-right">
          <span class="pill">2023-12 test</span>
          <span class="pill">cost 1 bps</span>
        </div>
      </div>

      <header id="overview" class="hero">
        <div>
          <div class="eyebrow">C Agent Controller</div>
          <h1>Trading Agent Console</h1>
        </div>
        <div class="hero-meta">
          <span class="pill">Forecast to action</span>
          <span class="pill">Execution aware</span>
          <span class="pill">RL feedback</span>
        </div>
      </header>

      <section class="section">
        <div class="section-head">
          <h2 class="section-title">Selected KPIs</h2>
          <div class="section-note">B Pearson 0.0677 / R2 0.00443</div>
        </div>
        <div class="metrics">{metrics}</div>
      </section>

      <section id="signal" class="section">
        <div class="section-head">
          <h2 class="section-title">Signal quality</h2>
          <div class="section-note">forecast rank vs realized fret12</div>
        </div>
        <div class="grid-2">
          <div class="panel">
            <div class="panel-head">
              <div class="panel-title">Forecast decile return</div>
              <div class="panel-kicker">bps</div>
            </div>
            <div class="panel-body chart">{decile_chart}</div>
          </div>
          <div class="panel">
            <div class="panel-head">
              <div class="panel-title">Policy spread</div>
              <div class="panel-kicker">long - short</div>
            </div>
            <div class="panel-body chart">{policy_chart}</div>
          </div>
        </div>
      </section>

      <section id="execution" class="section">
        <div class="section-head">
          <h2 class="section-title">Execution layer</h2>
          <div class="section-note">market / hybrid / limit</div>
        </div>
        <div class="grid-2">
          <div class="panel">
            <div class="panel-head">
              <div class="panel-title">Limit equity curve</div>
              <div class="panel-kicker">snapshot approximation</div>
            </div>
            <div class="panel-body chart">{equity_chart}</div>
          </div>
          <div class="panel">
            <div class="panel-head">
              <div class="panel-title">Final equity by order type</div>
              <div class="panel-kicker">initial 10,000,000</div>
            </div>
            <div class="panel-body">{execution_bars}</div>
          </div>
        </div>
      </section>

      <section id="factor" class="section">
        <div class="section-head">
          <h2 class="section-title">Factor mining</h2>
          <div class="section-note">best factor {best_factor}</div>
        </div>
        <div class="grid-2">
          <div class="panel">
            <div class="panel-head">
              <div class="panel-title">Top mined factors</div>
              <div class="panel-kicker">spread bps</div>
            </div>
            <div class="panel-body">{factor_rows}</div>
          </div>
          <div class="panel">
            <div class="panel-head">
              <div class="panel-title">Factor-aware diagnostic</div>
              <div class="panel-kicker">top 8 score</div>
            </div>
            <div class="panel-body">
              <div class="grid-3">
                {factor_metric_1}
                {factor_metric_2}
                {factor_metric_3}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="rl" class="section">
        <div class="section-head">
          <h2 class="section-title">RL selector</h2>
          <div class="section-note">contextual-bandit prototype</div>
        </div>
        <div class="grid-2">
          <div class="panel">
            <div class="panel-head">
              <div class="panel-title">Learned Q preferences</div>
              <div class="panel-kicker">highest values</div>
            </div>
            <div class="panel-body">{q_rows}</div>
          </div>
          <div class="panel">
            <div class="panel-head">
              <div class="panel-title">RL summary</div>
              <div class="panel-kicker">4,746 steps</div>
            </div>
            <div class="panel-body">
              <div class="grid-3">
                {rl_metric_1}
                {rl_metric_2}
                {rl_metric_3}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="decisions" class="section">
        <div class="section-head">
          <h2 class="section-title">Decision tape</h2>
          <div class="section-note">buy / sell / hold examples</div>
        </div>
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">Cross-sectional action samples</div>
            <div class="panel-kicker">expected and realized reward</div>
          </div>
          <div class="panel-body">{decision_table}</div>
        </div>
      </section>

      <section id="boundary" class="section">
        <div class="section-head">
          <h2 class="section-title">Boundary</h2>
          <div class="section-note">research prototype</div>
        </div>
        <div class="boundary">
          <div class="statement">
            <strong>What it is</strong>
            <span>Agent-assisted alpha mining plus a lightweight trading decision prototype.</span>
          </div>
          <div class="statement">
            <strong>What it is not</strong>
            <span>Not a production trading system or full exchange-level matching engine.</span>
          </div>
        </div>
      </section>
    </div>
  </main>
</div>
""".format(
        generated_at=esc(generated_at),
        metrics=metrics,
        decile_chart=svg_bar_chart(decile_labels, decile_values),
        policy_chart=svg_bar_chart(policy_labels, policy_values),
        equity_chart=svg_line_chart(equity_values),
        execution_bars=execution_bars(exec_rows),
        best_factor=esc(value(factor, "feature", "")),
        factor_rows=factor_rows(factors),
        factor_metric_1=metric_card("Factor spread", fmt_bps(value(factor_agent, "long_short_spread")), "diagnostic"),
        factor_metric_2=metric_card("Period hit", fmt_pct(value(factor_agent, "period_positive_spread_rate")), "factor score", "blue"),
        factor_metric_3=metric_card("Best IC", fmt_decimal(value(factor, "mean_ic"), digits=4), value(factor, "feature", ""), "red"),
        q_rows=q_rows(frames["rl_q"]),
        rl_metric_1=metric_card("Final equity", fmt_money(value(rl, "final_equity")), "RL selector"),
        rl_metric_2=metric_card("Reward hit", fmt_pct(value(rl, "positive_reward_rate")), "positive reward", "blue"),
        rl_metric_3=metric_card("Max drawdown", fmt_pct(value(rl, "max_drawdown")), "controlled", "red"),
        decision_table=decision_table(examples),
    )

    page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>C Agent Console</title>
  <style>{css}</style>
</head>
<body>
{body}
<script>{script}</script>
</body>
</html>
""".format(css=CSS, body=body, script=SCRIPT)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(page)
    return output_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a static dark GUI for the C Trading Agent dashboard.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--output-file", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(args.output_dir, "C_AGENT_GUI.html")
    path = build_html(args.output_dir, output_file)
    print("Saved GUI to {}".format(path))


if __name__ == "__main__":
    main()
