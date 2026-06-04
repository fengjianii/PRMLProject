import argparse
import html
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", ".."))
DEFAULT_DEPS = os.path.join(WORKSPACE_DIR, "tmp", "prml_pydeps")
if os.path.isdir(DEFAULT_DEPS) and DEFAULT_DEPS not in sys.path:
    sys.path.insert(0, DEFAULT_DEPS)

from agent_controller import (  # noqa: E402
    best_factor_row,
    first_row,
    fmt_bps,
    fmt_decimal,
    fmt_money,
    fmt_pct,
    read_outputs,
    select_row,
    value,
)
from agent_gui import (  # noqa: E402
    CSS,
    decision_table,
    execution_bars,
    execution_rows,
    factor_rows,
    fmt_money_compact,
    load_equity_curve,
    load_examples,
    metric_card,
    policy_rows,
    q_rows,
    svg_bar_chart,
    svg_line_chart,
)


OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
RUN_DIR = os.path.join(OUTPUT_DIR, "gui_runs")
PREDICTION_FILE = os.path.join("outputs", "B_prediction_output.csv")

JOBS = {}
JOB_LOCK = threading.Lock()
RUNNING_JOB_ID = None
MAX_LOG_CHARS = 18000


LIVE_CSS = r"""
.run-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.run-card {
  min-height: 112px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}

.run-card strong {
  display: block;
  margin-bottom: 7px;
  color: var(--text);
  font-size: 12px;
}

.run-card span {
  display: block;
  min-height: 34px;
  color: var(--muted);
  font-size: 11px;
  line-height: 1.55;
}

.run-button {
  width: 100%;
  height: 32px;
  margin-top: 12px;
  border: 1px solid #334155;
  border-radius: 6px;
  background: #121722;
  color: var(--text);
  font-size: 12px;
  font-weight: 800;
  cursor: pointer;
  transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
}

.run-button:hover {
  border-color: #3b82f6;
  background: #172033;
  transform: translateY(-1px);
}

.run-button:disabled {
  cursor: wait;
  opacity: 0.55;
  transform: none;
}

.console {
  min-height: 220px;
  max-height: 420px;
  overflow: auto;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #090b0f;
  color: #b7c3d7;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 11px;
  line-height: 1.55;
  white-space: pre-wrap;
}

.status-line {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  height: 26px;
  padding: 0 9px;
  border-radius: 6px;
  background: rgba(141, 150, 165, 0.14);
  color: var(--soft);
  font-size: 11px;
  font-weight: 850;
}

.status-badge.running {
  background: rgba(59, 130, 246, 0.16);
  color: #7cb2ff;
}

.status-badge.completed {
  background: rgba(34, 197, 94, 0.16);
  color: var(--green);
}

.status-badge.failed {
  background: rgba(240, 82, 82, 0.16);
  color: var(--red);
}

.mini-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.link-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 28px;
  padding: 0 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #101318;
  color: var(--soft);
  font-size: 12px;
  font-weight: 700;
}

@media (max-width: 1040px) {
  .run-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .run-grid {
    grid-template-columns: 1fr;
  }
}
"""


LIVE_SCRIPT = r"""
const buttons = Array.from(document.querySelectorAll("[data-task]"));
const statusBadge = document.querySelector("#run-status");
const logBox = document.querySelector("#run-log");
const jobTitle = document.querySelector("#job-title");
const reloadButton = document.querySelector("#reload-page");

function setButtons(disabled) {
  buttons.forEach((button) => {
    button.disabled = disabled;
  });
}

function setStatus(job) {
  if (!job) {
    statusBadge.className = "status-badge";
    statusBadge.textContent = "idle";
    jobTitle.textContent = "No job running";
    return;
  }
  statusBadge.className = `status-badge ${job.status}`;
  statusBadge.textContent = job.status;
  jobTitle.textContent = `${job.title} / ${job.id}`;
  logBox.textContent = job.log_tail || "";
  logBox.scrollTop = logBox.scrollHeight;
  setButtons(job.status === "running");
}

async function pollJob(id) {
  const response = await fetch(`/api/job?id=${encodeURIComponent(id)}`);
  if (!response.ok) {
    setButtons(false);
    return;
  }
  const job = await response.json();
  setStatus(job);
  if (job.status === "running") {
    setTimeout(() => pollJob(id), 1000);
  } else {
    setButtons(false);
  }
}

async function runTask(task) {
  setButtons(true);
  logBox.textContent = "Starting task...\n";
  const response = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task }),
  });
  const data = await response.json();
  if (!response.ok) {
    setButtons(false);
    logBox.textContent = data.error || "Failed to start task";
    return;
  }
  localStorage.setItem("agentLiveJobId", data.id);
  pollJob(data.id);
}

buttons.forEach((button) => {
  button.addEventListener("click", () => runTask(button.dataset.task));
});

reloadButton.addEventListener("click", () => window.location.reload());

const previousJobId = localStorage.getItem("agentLiveJobId");
if (previousJobId) {
  pollJob(previousJobId);
}
"""


TASKS = {
    "controller": {
        "title": "Refresh Controller",
        "description": "Regenerate dashboard and decision examples from current outputs.",
        "commands": lambda python: [
            [
                python,
                "agent_controller.py",
                "--predictions",
                PREDICTION_FILE,
                "--output-dir",
                "outputs",
            ],
        ],
    },
    "signal_backtest": {
        "title": "Run Signal Backtest",
        "description": "Rebuild policy comparison, deciles, period checks, and controller summary.",
        "commands": lambda python: [
            [
                python,
                "run_c_experiment.py",
                "--predictions",
                PREDICTION_FILE,
                "--output-dir",
                "outputs",
                "--cost-bps",
                "1",
            ],
            [
                python,
                "agent_controller.py",
                "--predictions",
                PREDICTION_FILE,
                "--output-dir",
                "outputs",
            ],
        ],
    },
    "rl_policy": {
        "title": "Run RL Selector",
        "description": "Run the Q-learning/contextual-bandit policy selector and refresh summary.",
        "commands": lambda python: [
            [
                python,
                "agent_rl_policy.py",
                "--predictions",
                PREDICTION_FILE,
                "--output-dir",
                "outputs",
                "--cost-bps",
                "1",
                "--alpha",
                "0.20",
                "--epsilon",
                "0.05",
            ],
            [
                python,
                "agent_controller.py",
                "--predictions",
                PREDICTION_FILE,
                "--output-dir",
                "outputs",
            ],
        ],
    },
    "execution_limit": {
        "title": "Run Limit Execution",
        "description": "Run the limit-order execution approximation and refresh controller output.",
        "commands": lambda python: [
            execution_command(python, "limit"),
            [
                python,
                "agent_controller.py",
                "--predictions",
                PREDICTION_FILE,
                "--output-dir",
                "outputs",
            ],
        ],
    },
    "execution_suite": {
        "title": "Run Execution Suite",
        "description": "Run market, hybrid, and limit execution simulations. This is slower.",
        "commands": lambda python: [
            execution_command(python, "market"),
            execution_command(python, "hybrid"),
            execution_command(python, "limit"),
            [
                python,
                "agent_controller.py",
                "--predictions",
                PREDICTION_FILE,
                "--output-dir",
                "outputs",
            ],
        ],
    },
    "factor_mining": {
        "title": "Mine Factors",
        "description": "Regenerate automatic factor mining report from h5 data. This is the heaviest task.",
        "commands": lambda python: [
            [
                python,
                "agent_factor_mining.py",
                "--data-dir",
                "data",
                "--start-date",
                "20231201",
                "--end-date",
                "20231229",
                "--output-dir",
                "outputs",
                "--top-k",
                "15",
                "--factor-agent-top-n",
                "8",
                "--cost-bps",
                "1",
            ],
            [
                python,
                "agent_controller.py",
                "--predictions",
                PREDICTION_FILE,
                "--output-dir",
                "outputs",
            ],
        ],
    },
    "static_gui": {
        "title": "Build Static HTML",
        "description": "Regenerate outputs/C_AGENT_GUI.html for offline presentation.",
        "commands": lambda python: [
            [python, "agent_gui.py", "--output-dir", "outputs"],
        ],
    },
    "compile_check": {
        "title": "Compile Check",
        "description": "Run py_compile on C-side scripts to catch syntax errors.",
        "commands": lambda python: [
            [
                python,
                "-m",
                "py_compile",
                "agent_gui_server.py",
                "agent_gui.py",
                "agent_controller.py",
                "agent_execution_sim.py",
                "agent_rl_policy.py",
                "agent_factor_mining.py",
                "agent_policy.py",
                "backtest.py",
                "run_c_experiment.py",
            ],
        ],
    },
}


def execution_command(python, style):
    return [
        python,
        "agent_execution_sim.py",
        "--predictions",
        PREDICTION_FILE,
        "--data-dir",
        "data",
        "--output-dir",
        "outputs",
        "--output-prefix",
        "C_execution_{}".format(style),
        "--order-style",
        style,
        "--notional-per-trade",
        "10000",
        "--max-gross-notional",
        "10000000",
        "--queue-ahead-frac",
        "0.50",
        "--impact-k",
        "2",
        "--fee-bps",
        "0.5",
        "--exit-impact-bps",
        "0.5",
    ]


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_float(raw, default=0.0):
    try:
        if raw is None:
            return default
        if hasattr(raw, "__float__"):
            value = float(raw)
        else:
            value = float(str(raw))
        if value != value:
            return default
        return value
    except (TypeError, ValueError):
        return default


def read_csv_if_exists(path):
    if not os.path.exists(path):
        return None
    import pandas as pd

    return pd.read_csv(path)


def latest_jobs(limit=6):
    with JOB_LOCK:
        items = list(JOBS.values())
    items.sort(key=lambda item: item["created_at"], reverse=True)
    return [public_job(item) for item in items[:limit]]


def public_job(job):
    return {
        "id": job["id"],
        "task": job["task"],
        "title": job["title"],
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "ended_at": job.get("ended_at"),
        "return_code": job.get("return_code"),
        "command": job.get("command_text", ""),
        "log_tail": job.get("log_tail", ""),
        "log_path": job.get("log_path", ""),
    }


def prepare_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if os.path.isdir(DEFAULT_DEPS):
        existing = env.get("PYTHONPATH", "")
        paths = [DEFAULT_DEPS]
        if existing:
            paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def start_job(task_name):
    global RUNNING_JOB_ID
    if task_name not in TASKS:
        raise ValueError("Unknown task: {}".format(task_name))
    with JOB_LOCK:
        if RUNNING_JOB_ID and JOBS.get(RUNNING_JOB_ID, {}).get("status") == "running":
            return None, "Another task is running: {}".format(RUNNING_JOB_ID)

        job_id = uuid.uuid4().hex[:10]
        os.makedirs(RUN_DIR, exist_ok=True)
        log_path = os.path.join(RUN_DIR, "{}_{}.log".format(
            datetime.now().strftime("%Y%m%d_%H%M%S"), task_name))
        job = {
            "id": job_id,
            "task": task_name,
            "title": TASKS[task_name]["title"],
            "status": "queued",
            "created_at": now_text(),
            "log_path": log_path,
            "log_tail": "",
        }
        JOBS[job_id] = job
        RUNNING_JOB_ID = job_id

    thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    thread.start()
    return public_job(job), None


def append_log(job, text):
    old = job.get("log_tail", "")
    combined = old + text
    if len(combined) > MAX_LOG_CHARS:
        combined = combined[-MAX_LOG_CHARS:]
    job["log_tail"] = combined


def run_job(job_id):
    global RUNNING_JOB_ID
    with JOB_LOCK:
        job = JOBS[job_id]
        job["status"] = "running"
        job["started_at"] = now_text()

    python = sys.executable
    task = TASKS[job["task"]]
    commands = task["commands"](python)
    env = prepare_env()
    return_code = 0

    try:
        with open(job["log_path"], "w", encoding="utf-8") as log_file:
            log_file.write("[{}] {}\n".format(now_text(), job["title"]))
            for index, command in enumerate(commands, 1):
                command_text = " ".join(command)
                with JOB_LOCK:
                    job["command_text"] = command_text
                    append_log(job, "\n$ {}\n".format(command_text))
                log_file.write("\n$ {}\n".format(command_text))
                log_file.flush()

                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_DIR,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert process.stdout is not None
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    with JOB_LOCK:
                        append_log(job, line)
                process.wait()
                return_code = process.returncode
                end_line = "[command {}/{} exited {}]\n".format(
                    index, len(commands), return_code)
                log_file.write(end_line)
                with JOB_LOCK:
                    append_log(job, end_line)
                if return_code != 0:
                    break
    except Exception:
        return_code = -1
        error = traceback.format_exc()
        with JOB_LOCK:
            append_log(job, error)
    finally:
        with JOB_LOCK:
            job["return_code"] = return_code
            job["ended_at"] = now_text()
            job["status"] = "completed" if return_code == 0 else "failed"
            if RUNNING_JOB_ID == job_id:
                RUNNING_JOB_ID = None


def run_cards_html():
    cards = []
    order = [
        "controller",
        "signal_backtest",
        "rl_policy",
        "execution_limit",
        "execution_suite",
        "factor_mining",
        "static_gui",
        "compile_check",
    ]
    for task_name in order:
        task = TASKS[task_name]
        cards.append("""
<div class="run-card">
  <strong>{title}</strong>
  <span>{description}</span>
  <button class="run-button" data-task="{task_name}">Run</button>
</div>
""".format(
            title=html.escape(task["title"]),
            description=html.escape(task["description"]),
            task_name=html.escape(task_name),
        ))
    return "\n".join(cards)


def build_live_dashboard():
    frames = read_outputs("outputs")
    examples = load_examples("outputs")
    equity = load_equity_curve("outputs")

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
    policy_labels = [label_map.get(item["label"], item["label"]) for item in policies]
    policy_values = [item["spread_bps"] for item in policies]

    equity_values = []
    if equity is not None and "equity" in equity.columns:
        equity_values = equity["equity"].tolist()

    metrics = "\n".join([
        metric_card("Signal spread", fmt_bps(value(policy, "long_short_spread")),
                    "top-bottom 10%"),
        metric_card("Positive periods", fmt_pct(value(policy, "period_positive_spread_rate")),
                    "4,746 periods"),
        metric_card("Best policy", fmt_bps(value(top5, "long_short_spread")),
                    "top-bottom 5/5", "blue"),
        metric_card("Limit equity", fmt_money_compact(value(exec_limit, "final_equity")),
                    "fill {}".format(fmt_pct(value(exec_limit, "fill_rate")))),
        metric_card("RL equity", fmt_money_compact(value(rl, "final_equity")),
                    "reward {}".format(fmt_pct(value(rl, "positive_reward_rate")))),
    ])

    prediction_ok = os.path.exists(os.path.join(PROJECT_DIR, PREDICTION_FILE))
    data_ok = os.path.isdir(os.path.join(PROJECT_DIR, "data"))
    latest = latest_jobs()
    latest_text = "No jobs yet"
    if latest:
        latest_text = "{} / {} / {}".format(
            latest[0]["title"], latest[0]["status"], latest[0]["created_at"])

    return """
<div class="app">
  <aside class="sidebar">
    <div class="brand"><span class="brand-mark">C</span><span>MEOW Agent</span></div>
    <a class="back" href="#run">Run code</a>
    <div class="nav-title">Live Console</div>
    <a class="nav-link active" href="#run"><span class="nav-dot"></span>Run</a>
    <a class="nav-link" href="#overview"><span class="nav-dot"></span>Overview</a>
    <a class="nav-link" href="#signal"><span class="nav-dot"></span>Signal</a>
    <a class="nav-link" href="#execution"><span class="nav-dot"></span>Execution</a>
    <a class="nav-link" href="#decisions"><span class="nav-dot"></span>Decisions</a>
    <div class="nav-title">Research</div>
    <a class="nav-link" href="#factor"><span class="nav-dot"></span>Factor mining</a>
    <a class="nav-link" href="#rl"><span class="nav-dot"></span>RL selector</a>
    <a class="nav-link" href="#boundary"><span class="nav-dot"></span>Boundary</a>
    <div class="sidebar-foot">
      <div>Branch: c-agent-backtest</div>
      <div>Prediction: {prediction_state}</div>
      <div>Data dir: {data_state}</div>
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
          <span class="pill">live server</span>
          <span class="pill">127.0.0.1</span>
        </div>
      </div>

      <header id="overview" class="hero">
        <div>
          <div class="eyebrow">C Agent Controller</div>
          <h1>Trading Agent Live Console</h1>
        </div>
        <div class="hero-meta">
          <span class="pill">Run Python</span>
          <span class="pill">Refresh outputs</span>
          <span class="pill">Inspect logs</span>
        </div>
      </header>

      <section id="run" class="section">
        <div class="section-head">
          <h2 class="section-title">Run code</h2>
          <div class="section-note">latest: {latest_text}</div>
        </div>
        <div class="run-grid">{run_cards}</div>
      </section>

      <section class="section">
        <div class="status-line">
          <div>
            <h2 class="section-title" id="job-title">No job running</h2>
          </div>
          <div class="mini-actions">
            <span id="run-status" class="status-badge">idle</span>
            <button id="reload-page" class="link-button" type="button">Reload metrics</button>
            <a class="link-button" href="/outputs/C_AGENT_GUI.html" target="_blank">Static GUI</a>
          </div>
        </div>
        <pre id="run-log" class="console">Click a Run button to execute project code. Logs will appear here.</pre>
      </section>

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
            <div class="panel-head"><div class="panel-title">Forecast decile return</div><div class="panel-kicker">bps</div></div>
            <div class="panel-body chart">{decile_chart}</div>
          </div>
          <div class="panel">
            <div class="panel-head"><div class="panel-title">Policy spread</div><div class="panel-kicker">long - short</div></div>
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
            <div class="panel-head"><div class="panel-title">Limit equity curve</div><div class="panel-kicker">snapshot approximation</div></div>
            <div class="panel-body chart">{equity_chart}</div>
          </div>
          <div class="panel">
            <div class="panel-head"><div class="panel-title">Final equity by order type</div><div class="panel-kicker">initial 10,000,000</div></div>
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
            <div class="panel-head"><div class="panel-title">Top mined factors</div><div class="panel-kicker">spread bps</div></div>
            <div class="panel-body">{factor_rows}</div>
          </div>
          <div class="panel">
            <div class="panel-head"><div class="panel-title">Factor-aware diagnostic</div><div class="panel-kicker">top 8 score</div></div>
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
            <div class="panel-head"><div class="panel-title">Learned Q preferences</div><div class="panel-kicker">highest values</div></div>
            <div class="panel-body">{q_rows}</div>
          </div>
          <div class="panel">
            <div class="panel-head"><div class="panel-title">RL summary</div><div class="panel-kicker">4,746 steps</div></div>
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
          <div class="panel-head"><div class="panel-title">Cross-sectional action samples</div><div class="panel-kicker">expected and realized reward</div></div>
          <div class="panel-body">{decision_table}</div>
        </div>
      </section>

      <section id="boundary" class="section">
        <div class="section-head">
          <h2 class="section-title">Boundary</h2>
          <div class="section-note">local research runner</div>
        </div>
        <div class="boundary">
          <div class="statement">
            <strong>What it runs</strong>
            <span>Only fixed project scripts from an allowlist. The browser cannot submit arbitrary shell commands.</span>
          </div>
          <div class="statement">
            <strong>What it is not</strong>
            <span>Not a production trading UI. It is a local demo runner for experiments and presentation.</span>
          </div>
        </div>
      </section>
    </div>
  </main>
</div>
""".format(
        prediction_state="ready" if prediction_ok else "missing",
        data_state="ready" if data_ok else "missing",
        latest_text=html.escape(latest_text),
        run_cards=run_cards_html(),
        metrics=metrics,
        decile_chart=svg_bar_chart(decile_labels, decile_values),
        policy_chart=svg_bar_chart(policy_labels, policy_values),
        equity_chart=svg_line_chart(equity_values),
        execution_bars=execution_bars(execution_rows(frames)),
        best_factor=html.escape(str(value(factor, "feature", ""))),
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


def build_page():
    body = build_live_dashboard()
    css = CSS + "\n" + LIVE_CSS
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>C Agent Live Console</title>
  <style>{css}</style>
</head>
<body>
{body}
<script>{script}</script>
</body>
</html>
""".format(css=css, body=body, script=LIVE_SCRIPT)


class AgentGuiHandler(BaseHTTPRequestHandler):
    server_version = "AgentGuiServer/1.0"

    def log_message(self, fmt, *args):
        return

    def send_json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_html(self, text, status=200):
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(build_page())
            return
        if parsed.path == "/api/jobs":
            self.send_json({"jobs": latest_jobs(limit=12)})
            return
        if parsed.path == "/api/job":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            with JOB_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self.send_json({"error": "Job not found"}, status=404)
                return
            self.send_json(public_job(job))
            return
        if parsed.path.startswith("/outputs/"):
            self.serve_output_file(parsed.path[len("/outputs/"):])
            return
        self.send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_json({"error": "Not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            data = json.loads(body)
            task_name = data.get("task", "")
            job, error = start_job(task_name)
            if error:
                self.send_json({"error": error}, status=409)
                return
            self.send_json(job)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def serve_output_file(self, relative_path):
        safe_rel = unquote(relative_path).replace("\\", "/")
        if safe_rel.startswith("../") or "/../" in safe_rel or safe_rel == "..":
            self.send_json({"error": "Invalid path"}, status=400)
            return
        path = os.path.abspath(os.path.join(OUTPUT_DIR, safe_rel))
        if not path.startswith(os.path.abspath(OUTPUT_DIR)):
            self.send_json({"error": "Invalid path"}, status=400)
            return
        if not os.path.exists(path) or not os.path.isfile(path):
            self.send_json({"error": "File not found"}, status=404)
            return
        with open(path, "rb") as f:
            payload = f.read()
        content_type = "text/plain; charset=utf-8"
        if path.endswith(".html"):
            content_type = "text/html; charset=utf-8"
        elif path.endswith(".svg"):
            content_type = "image/svg+xml"
        elif path.endswith(".png"):
            content_type = "image/png"
        elif path.endswith(".csv"):
            content_type = "text/csv; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local live GUI server for the C Trading Agent.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), AgentGuiHandler)
    print("C Agent Live Console: http://{}:{}/".format(args.host, args.port))
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
