#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
monitor.py — Genesis Architecture unified monitoring dashboard

Textual TUI that supersedes throttle.py for day-to-day use.
throttle.py remains available as a minimal ANSI fallback.

Layout
  ┌─ Header + clock ────────────────────────────────────────────────┐
  │ Left 52 cols: sys stats / processes / vite apps                 │
  │ Right 1fr:    pipeline progress / controller / GPU              │
  ├─────────────────────────────────────────────────────────────────┤
  │ Logs (tabbed: Indexer | Graph | Query)                height 12 │
  └─ Help bar ──────────────────────────────────────────────────────┘

Controls
  ↑/↓       navigate rows            M  toggle AUTO/MANUAL
  ←/→       adjust reserve or cap    A  cycle CPU affinity
  R R        reboot selected process  P  cycle OS priority
  K K        kill selected process    V  launch/stop Vite app
  Q          quit (resumes all suspended processes via atexit)

Run:     python monitor.py
Fallback: python throttle.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import psutil
from rich.markup import escape as _esc

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, RichLog, Static, TabbedContent, TabPane

# shared CPU-cap engine and process helpers
import throttle as _t

warnings.filterwarnings("ignore", category=FutureWarning)

_HERE        = Path(__file__).parent
_LOGS_DIR    = _HERE / "logs"
_GRAPH_STATE = _HERE / "graph_state.json"

# ── GPU via pynvml ────────────────────────────────────────────────────────────

_nvml_ok     = False
_nvml_handle = None
try:
    import pynvml as _nvml  # type: ignore
    _nvml.nvmlInit()
    _nvml_handle = _nvml.nvmlDeviceGetHandleByIndex(0)
    _nvml_ok     = True
except Exception:
    pass


def _gpu() -> dict:
    if not _nvml_ok:
        return {}
    try:
        u = _nvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
        m = _nvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        t = _nvml.nvmlDeviceGetTemperature(
            _nvml_handle, _nvml.NVML_TEMPERATURE_GPU
        )
        return {
            "pct":   u.gpu,
            "used":  m.used  >> 20,
            "total": m.total >> 20,
            "temp":  t,
        }
    except Exception:
        return {}


# ── helpers ───────────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _or_reset() -> str:
    """HH:MM:SS until OpenRouter daily rate-limit resets (midnight UTC)."""
    now      = datetime.now(timezone.utc)
    midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    secs = int((midnight - now).total_seconds()) % 86400
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _graph_counts() -> dict:
    try:
        data = json.loads(
            _GRAPH_STATE.read_bytes().decode("utf-8", errors="replace")
        )
        counts: dict = {}
        for v in data.get("files", {}).values():
            st = v.get("status", "unknown")
            counts[st] = counts.get(st, 0) + 1
        return counts
    except Exception:
        return {}


def _bar(pct: float, w: int = 16) -> str:
    n = max(0, min(w, round(pct / 100 * w)))
    return "█" * n + "░" * (w - n)


def _cbar(pct: float, w: int = 16) -> str:
    c = "red" if pct >= 75 else ("yellow" if pct >= 45 else "green")
    return f"[{c}]{_bar(pct, w)}[/]"


def _mem(mb: float) -> str:
    return f"{mb / 1024:.1f}G" if mb >= 1024 else f"{int(mb)}M"


# ── log tailing ───────────────────────────────────────────────────────────────

_LOG_INIT_BYTES = 8_000
_log_pos: dict[str, int] = {}


def _log_seed(path: Path) -> list[str]:
    """Return last ~60 lines for initial display, advance position marker."""
    try:
        sz    = path.stat().st_size
        start = max(0, sz - _LOG_INIT_BYTES)
        with path.open("rb") as f:
            f.seek(start)
            raw = f.read()
            _log_pos[str(path)] = f.tell()
        lines = raw.decode("utf-8", errors="replace").splitlines()
        return (lines[1:] if start > 0 else lines)[-60:]
    except OSError:
        return []


def _log_new(path: Path) -> list[str]:
    """Lines appended since last call."""
    key = str(path)
    try:
        sz  = path.stat().st_size
        pos = _log_pos.get(key, max(0, sz - _LOG_INIT_BYTES))
        if sz <= pos:
            return []
        with path.open("rb") as f:
            f.seek(pos)
            raw = f.read()
            _log_pos[key] = f.tell()
        return raw.decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []


# ── log file discovery ────────────────────────────────────────────────────────

# (path, widget-id, tab-title)
_LOG_TABS = [
    (_LOGS_DIR / "indexer.log",          "rl_indexer", "Indexer"),
    (_LOGS_DIR / "graph_builder.log",    "rl_graph",   "Graph"),
    (_HERE      / "query.out.log",        "rl_query",   "Query"),
]


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
Screen { background: $background; }

#main {
    height: 1fr;
    min-height: 18;
}

#left {
    width: 54;
    border-right: tall $panel;
    padding: 0 1;
    overflow-y: auto;
}

#right {
    width: 1fr;
    padding: 0 1;
    overflow-y: auto;
}

#logs {
    height: 12;
    border-top: tall $panel;
}

#help {
    height: 1;
    background: $panel;
    padding: 0 1;
    color: $text-muted;
}

Static { background: $background; }
RichLog { background: $background; scrollbar-gutter: stable; }
TabPane { padding: 0 1; }
"""


# ── App ───────────────────────────────────────────────────────────────────────

class MonitorApp(App):
    CSS   = _CSS
    TITLE = "Genesis Architecture — Monitor"

    BINDINGS = [
        Binding("q",     "app.quit",     "Quit",     show=False),
        Binding("m",     "toggle_mode",  "Mode",     show=False),
        Binding("up",    "nav_up",       "Up",       show=False),
        Binding("down",  "nav_down",     "Down",     show=False),
        Binding("left",  "adj_left",     "Left",     show=False),
        Binding("right", "adj_right",    "Right",    show=False),
        Binding("a",     "cycle_aff",    "Affinity", show=False),
        Binding("p",     "cycle_prio",   "Priority", show=False),
        Binding("r",     "reboot_key",   "Reboot",   show=False),
        Binding("k",     "kill_key",     "Kill",     show=False),
        Binding("v",     "vite_key",     "Vite",     show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._sel:      int            = 0
        self._tasks:    list           = []
        self._vapps:    list           = []
        self._cpus:     dict           = {}
        self._sys_cpu:  float          = 0.0
        self._status:   str            = ""
        self._kill_pid: Optional[int]  = None
        self._kill_ts:  float          = 0.0
        self._rb_ts:    dict[int, float]  = {}   # pid  → confirm timestamp
        self._vt_ts:    dict[str, float]  = {}   # name → confirm timestamp

    # ── compose ───────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Static("", id="left_panel", markup=True)
            with Vertical(id="right"):
                yield Static("", id="right_panel", markup=True)
        with TabbedContent(id="logs"):
            for path, wid, title in _LOG_TABS:
                with TabPane(title, id=f"pane_{wid}"):
                    yield RichLog(
                        id=wid,
                        highlight=False,
                        markup=False,
                        max_lines=500,
                    )
        yield Static(
            "[dim]↑↓[/] nav  "
            "[dim]M[/] mode  "
            "[dim]←→[/] adj  "
            "[cyan]A[/] aff  "
            "[cyan]P[/] pri  "
            "[yellow]R R[/] reboot  "
            "[red]K K[/] kill  "
            "[cyan]V[/] vite  "
            "[dim]Q[/] quit",
            id="help",
            markup=True,
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        psutil.cpu_percent(interval=None)   # prime non-blocking sampler
        self._seed_logs()
        self.set_interval(2.0, self._tick)

    def _seed_logs(self) -> None:
        for path, wid, _ in _LOG_TABS:
            lines = _log_seed(path)
            if not lines:
                continue
            try:
                rl = self.query_one(f"#{wid}", RichLog)
                for ln in lines:
                    rl.write(ln)
                rl.scroll_end(animate=False)
            except Exception:
                pass

    # ── main 2-second tick ────────────────────────────────────────────────────

    def _tick(self) -> None:
        now = time.time()

        # expire double-press confirms
        if self._kill_pid and now - self._kill_ts > 3:
            self._kill_pid, self._status = None, ""
        for k in [k for k, ts in list(self._rb_ts.items()) if now - ts > 3]:
            del self._rb_ts[k]
        for k in [k for k, ts in list(self._vt_ts.items()) if now - ts > 3]:
            del self._vt_ts[k]

        # scan live processes
        self._tasks = _t.get_tasks()
        cfg         = _t._cfg()
        self._vapps = cfg.get("vite_apps", _t._DEFAULT_CONFIG["vite_apps"])
        total       = len(self._tasks) + len(self._vapps)
        self._sel   = min(self._sel, max(0, total - 1))

        # CPU sampling (non-blocking second-call pattern)
        self._sys_cpu = psutil.cpu_percent(interval=None)
        for p in self._tasks:
            try:
                p.cpu_percent(interval=None)
            except Exception:
                pass

        # clean up caps for dead processes
        live = {p.pid for p in self._tasks}
        for dead in [pid for pid in list(_t._cap_threads) if pid not in live]:
            _t.remove_cap(dead)

        self._cpus = {}
        for p in self._tasks:
            try:
                self._cpus[p.pid] = p.cpu_percent(interval=None)
            except Exception:
                self._cpus[p.pid] = 0.0

        # run P-controller
        idx_pids = {p.pid for p in self._tasks if _t.is_indexer(p)}
        idx_cpu  = sum(self._cpus.get(pid, 0.0) for pid in idx_pids)
        _t.update_controller(idx_pids, idx_cpu, self._sys_cpu)

        self._render_left()
        self._render_right()
        self._update_logs()

    # ── left panel ────────────────────────────────────────────────────────────

    def _render_left(self) -> None:
        ram     = psutil.virtual_memory()
        ram_u   = ram.used  >> 20
        ram_t   = ram.total >> 20
        cpu_c   = "red" if self._sys_cpu >= 80 else (
                  "yellow" if self._sys_cpu >= 50 else "green")

        lines: list[str] = [
            f"[bold]SYS[/]  CPU {_cbar(self._sys_cpu, 10)} "
            f"[{cpu_c}]{self._sys_cpu:4.1f}%[/]  "
            f"RAM [dim]{_mem(ram_u)}/{_mem(ram_t)}[/]",
            f"[dim]{'─' * 50}[/]",
            "[bold]PROCESSES[/]",
            f"[dim]{'SCRIPT':<22} {'PID':>6}  {'CPU':>5}  {'MEM':>6}  CAP[/]",
            f"[dim]{'─' * 50}[/]",
        ]

        if not self._tasks:
            lines.append("[dim](no pipeline processes running)[/]")
        else:
            for i, p in enumerate(self._tasks):
                is_sel = i == self._sel
                try:
                    lbl    = _esc(_t.task_label(p)[:21])
                    pct    = self._cpus.get(p.pid, 0.0)
                    mem_mb = p.memory_info().rss >> 20
                    cap    = _t.get_cap(p.pid)
                    tag    = "[cyan][I][/]" if _t.is_indexer(p) else "[blue][S][/]"
                    cc     = "red" if pct >= 80 else ("yellow" if pct >= 40 else "green")
                    rbb    = " [yellow]↺[/]" if p.pid in self._rb_ts else "  "
                    row    = (
                        f"{tag} {lbl:<21} {p.pid:>6}  "
                        f"[{cc}]{pct:5.1f}%[/]  "
                        f"[dim]{_mem(mem_mb):>6}[/]  "
                        f"[magenta]{cap:>3}%[/]{rbb}"
                    )
                    lines.append(f"[reverse]{row}[/reverse]" if is_sel else row)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    lines.append("[dim](process gone)[/]")

        lines += [f"[dim]{'─' * 50}[/]", "[bold]VITE APPS[/]"]
        n = len(self._tasks)
        for j, app in enumerate(self._vapps):
            is_sel  = (n + j) == self._sel
            running, pid = _t.vite_probe(app["port"])
            cfm     = app["name"] in self._vt_ts
            if running:
                st = f"[green]● RUNNING[/] PID {pid}"
                if cfm:
                    st += "  [yellow]V again to stop[/]"
            else:
                st = "[dim]○ stopped[/]"
            row = f"  {_esc(app['name']):<14} :{app['port']}  {st}"
            lines.append(f"[reverse]{row}[/reverse]" if is_sel else row)

        if self._status:
            lines += [f"[dim]{'─' * 50}[/]", self._status]

        self.query_one("#left_panel", Static).update("\n".join(lines))

    # ── right panel ───────────────────────────────────────────────────────────

    def _render_right(self) -> None:
        indexed, total, fph, eta = _t.read_progress()
        g       = _graph_counts()
        g_done  = g.get("complete", 0)
        g_err   = g.get("error", 0)
        pct_idx = (indexed / total   * 100) if total   > 0 else 0.0
        pct_g   = (g_done  / indexed * 100) if indexed > 0 else 0.0

        with _t._ctrl_lock:
            mode    = _t._mode
            reserve = _t._reserve
            cap     = round(_t._cap_ema)
            man_cap = _t._manual_cap
            amb     = _t._ambient_ema

        lines: list[str] = [
            "[bold]PIPELINE[/]",
            f"[dim]{'─' * 40}[/]",
            (f"  Indexed  [cyan]{indexed:>4}[/]/{total:<5}  "
             f"{_cbar(pct_idx)} {pct_idx:4.1f}%"),
            (f"  Graph    [cyan]{g_done:>4}[/]/{indexed:<5}  "
             f"{_cbar(pct_g)} {pct_g:4.1f}%"),
            f"  Failed   [red]{g_err}[/]",
            f"  ETA      [cyan]{eta}[/]",
            f"  OR reset [yellow]{_or_reset()}[/]  [dim](50 req/day)[/]",
            f"[dim]{'─' * 40}[/]",
            "[bold]CONTROLLER[/]",
            f"  Mode     [{'cyan' if mode == 'auto' else 'yellow'}]{mode.upper()}[/]  [dim](M)[/]",
            f"  Ambient  {_cbar(amb)} [dim]{amb:4.0f}%[/]",
        ]

        if mode == "auto":
            lines += [
                f"  Reserve  {_cbar(reserve)} [yellow]{reserve}%[/]  [dim](←→)[/]",
                f"  Cap      {_cbar(cap)} [magenta]{cap}%[/]",
            ]
        else:
            lines += [
                f"  Cap      {_cbar(man_cap)} [yellow]{man_cap}%[/]  [dim](←→)[/]",
            ]

        gpu = _gpu()
        if gpu:
            gc = "red" if gpu["pct"] >= 90 else ("yellow" if gpu["pct"] >= 60 else "green")
            vram_pct = gpu["used"] / max(gpu["total"], 1) * 100
            lines += [
                f"[dim]{'─' * 40}[/]",
                "[bold]GPU — GTX 1650[/]",
                (f"  Util     {_cbar(gpu['pct'])} [{gc}]{gpu['pct']:3d}%[/]  "
                 f"[dim]{gpu['temp']}°C[/]"),
                (f"  VRAM     {_cbar(vram_pct)} "
                 f"[dim]{gpu['used']:4d}/{gpu['total']}M[/]"),
            ]

        self.query_one("#right_panel", Static).update("\n".join(lines))

    # ── log updates ───────────────────────────────────────────────────────────

    def _update_logs(self) -> None:
        for path, wid, _ in _LOG_TABS:
            try:
                rl = self.query_one(f"#{wid}", RichLog)
                for ln in _log_new(path):
                    if ln.strip():
                        rl.write(ln)
            except Exception:
                pass

    # ── keyboard actions ──────────────────────────────────────────────────────

    def _sel_task(self):
        if 0 <= self._sel < len(self._tasks):
            return self._tasks[self._sel]
        return None

    def _sel_vite(self):
        vi = self._sel - len(self._tasks)
        if 0 <= vi < len(self._vapps):
            return self._vapps[vi]
        return None

    def action_nav_up(self) -> None:
        self._sel = max(0, self._sel - 1)
        self._tick()

    def action_nav_down(self) -> None:
        mx = len(self._tasks) + len(self._vapps) - 1
        self._sel = min(max(mx, 0), self._sel + 1)
        self._tick()

    def action_toggle_mode(self) -> None:
        with _t._ctrl_lock:
            _t._mode = "manual" if _t._mode == "auto" else "auto"
        self._tick()

    def action_adj_left(self) -> None:
        with _t._ctrl_lock:
            if _t._mode == "auto":
                _t._reserve    = max(10, _t._reserve   - 10)
            else:
                _t._manual_cap = max(10, _t._manual_cap - 10)
        self._tick()

    def action_adj_right(self) -> None:
        with _t._ctrl_lock:
            if _t._mode == "auto":
                _t._reserve    = min(90, _t._reserve   + 10)
            else:
                _t._manual_cap = min(90, _t._manual_cap + 10)
        self._tick()

    def action_cycle_aff(self) -> None:
        t = self._sel_task()
        if t:
            _t.cycle_affinity(t)
            self._tick()

    def action_cycle_prio(self) -> None:
        t = self._sel_task()
        if t:
            _t.cycle_prio(t)
            self._tick()

    def action_reboot_key(self) -> None:
        t = self._sel_task()
        if not t:
            return
        now = time.time()
        pid = t.pid
        if pid in self._rb_ts and now - self._rb_ts[pid] < 3:
            del self._rb_ts[pid]
            self._status = _esc(_strip_ansi(_t.reboot_task(t)))
        else:
            self._rb_ts[pid] = now
            self._status = f"[yellow]R again within 3s to reboot PID {pid}[/]"
        self._tick()

    def action_kill_key(self) -> None:
        t = self._sel_task()
        if not t:
            return
        now = time.time()
        if self._kill_pid == t.pid and now - self._kill_ts < 3:
            try:
                lbl = _esc(_t.task_label(t))
                _t.remove_cap(t.pid)
                t.terminate()
                self._status = f"[red]SIGTERM → {lbl} (PID {t.pid})[/]"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._status = "[yellow]Already gone[/]"
            self._kill_pid = None
        else:
            lbl = _esc(_t.task_label(t))
            self._kill_pid, self._kill_ts = t.pid, now
            self._status = (
                f"[yellow]K again within 3s to kill {lbl} (PID {t.pid})[/]"
            )
        self._tick()

    def action_vite_key(self) -> None:
        app = self._sel_vite()
        if not app:
            return
        now  = time.time()
        name = app["name"]
        running, _ = _t.vite_probe(app["port"])
        if running:
            if name in self._vt_ts and now - self._vt_ts[name] < 3:
                del self._vt_ts[name]
                self._status = _esc(_strip_ansi(_t.kill_vite(app["port"])))
            else:
                self._vt_ts[name] = now
                self._status = f"[yellow]V again to stop {_esc(name)}[/]"
        else:
            self._status = _esc(_strip_ansi(_t.launch_vite(app)))
        self._tick()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    MonitorApp().run()
