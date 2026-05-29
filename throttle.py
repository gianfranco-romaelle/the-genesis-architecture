#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
throttle.py — adaptive throughput controller + service launcher

Reads throttle_config.json every 2 s (hot-reload) — add a new script
name and it appears in the display the next cycle without restarting.

Controls  (↑/↓ navigates a unified list: pipeline tasks + Vite apps)
  M       toggle AUTO / MANUAL mode
  ←/→     AUTO: adjust reserve %  |  MANUAL: adjust fixed cap %  (10 pp)
  ── pipeline task selected ──────────────────────────────────────────
  A       cycle CPU affinity  (all → half → quarter → 2 → 1 → all)
  P       cycle OS priority   (nudge on top of duty-cycle cap)
  R R     reboot  (kill + relaunch same command, two presses within 3 s)
  K K     kill  (two presses within 3 s)
  ── Vite app selected ───────────────────────────────────────────────
  V       launch if stopped  |  V V to kill if running
  Q       quit  (resumes all suspended processes before exit)
"""
import sys, json, time, re, subprocess, threading, atexit, msvcrt, psutil
from pathlib import Path
from datetime import datetime

_HERE           = Path(__file__).parent
_CONFIG_PATH    = _HERE / "throttle_config.json"
_PIPELINE_STATE = _HERE / "pipeline_state.json"
_LOGS_DIR       = _HERE / "logs"

_DEFAULT_CONFIG: dict = {
    "indexers": ["build_index.py", "index_timeline.py", "graph_builder.py"],
    "servers":  ["query.py"],
    "vite_apps": [
        {"name": "constellation", "cwd": "constellation", "port": 5173, "script": "dev"},
        {"name": "scriptorium",   "cwd": "scriptorium",   "port": 5174, "script": "dev"},
    ],
}

# ── config hot-reload ─────────────────────────────────────────────────────────

_config:       dict  = dict(_DEFAULT_CONFIG)
_config_mtime: float = 0.0
_config_lock         = threading.Lock()


def _load_config() -> dict:
    global _config, _config_mtime
    try:
        mtime = _CONFIG_PATH.stat().st_mtime
        if mtime != _config_mtime:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            with _config_lock:
                _config       = data
                _config_mtime = mtime
    except (OSError, json.JSONDecodeError):
        pass
    with _config_lock:
        return dict(_config)


def _cfg() -> dict:
    return _load_config()


# ── Windows priority classes ──────────────────────────────────────────────────

_WIN_PRIORITIES = [
    (psutil.IDLE_PRIORITY_CLASS,         "IDLE"),
    (psutil.BELOW_NORMAL_PRIORITY_CLASS, "BLW-NORM"),
    (psutil.NORMAL_PRIORITY_CLASS,       "NORMAL"),
    (psutil.ABOVE_NORMAL_PRIORITY_CLASS, "ABV-NORM"),
    (psutil.HIGH_PRIORITY_CLASS,         "HIGH"),
]
_PRIO_VALUES = [v for v, _ in _WIN_PRIORITIES]
_PRIO_LABELS = {v: lbl for v, lbl in _WIN_PRIORITIES}

# ── controller state ──────────────────────────────────────────────────────────

_mode        = "auto"
_reserve     = 40       # AUTO: % to keep free for user
_manual_cap  = 50       # MANUAL: fixed cap %
_ambient_ema = 30.0
_cap_ema     = 50.0
_ctrl_lock   = threading.Lock()

# ── suspend/resume cap engine ─────────────────────────────────────────────────

_cap_threads: dict[int, tuple] = {}
_caps:        dict[int, int]   = {}
_caps_lock  = threading.Lock()
CAP_PERIOD  = 0.10


def _cap_worker(pid: int, stop: threading.Event):
    while not stop.is_set():
        with _caps_lock:
            pct = _caps.get(pid, 100)
        if pct >= 100:
            try:
                psutil.Process(pid).resume()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            stop.wait(0.3)
            continue
        run_t = CAP_PERIOD * pct / 100
        off_t = CAP_PERIOD * (1 - pct / 100)
        try:
            p = psutil.Process(pid)
            p.resume()
            time.sleep(run_t)
            p.suspend()
            time.sleep(off_t)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
    try:
        psutil.Process(pid).resume()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def set_cap(pid: int, pct: int):
    pct = max(10, min(pct, 100))
    with _caps_lock:
        _caps[pid] = pct
    if pct >= 100:
        if pid in _cap_threads:
            _, stop = _cap_threads.pop(pid)
            stop.set()
        return
    if pid not in _cap_threads or not _cap_threads[pid][0].is_alive():
        stop = threading.Event()
        t    = threading.Thread(target=_cap_worker, args=(pid, stop),
                                daemon=True, name=f"cap-{pid}")
        _cap_threads[pid] = (t, stop)
        t.start()


def remove_cap(pid: int):
    set_cap(pid, 100)


def get_cap(pid: int) -> int:
    with _caps_lock:
        return _caps.get(pid, 100)


@atexit.register
def _cleanup_caps():
    for _, (_, stop) in list(_cap_threads.items()):
        stop.set()
    time.sleep(CAP_PERIOD * 3)
    for pid in list(_cap_threads):
        try:
            psutil.Process(pid).resume()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


# ── process helpers ───────────────────────────────────────────────────────────

def cpu_count() -> int:
    return psutil.cpu_count(logical=True) or 1


def _script_of(p) -> str:
    """First non-flag .py arg in cmdline. Never matches -c/-m script bodies."""
    try:
        info  = getattr(p, "info", None)
        parts = (info.get("cmdline") if isinstance(info, dict) else None) or p.cmdline()
        for arg in parts[1:]:
            if arg in ("-c", "-m"):
                break
            if arg.startswith("-"):
                continue
            if arg.endswith(".py"):
                return arg
            break
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError, TypeError, OSError):
        pass
    return ""


def is_indexer(p) -> bool:
    s = _script_of(p)
    return any(pat in s for pat in _cfg().get("indexers", _DEFAULT_CONFIG["indexers"]))


def get_tasks() -> list:
    patterns = (_cfg().get("indexers", []) + _cfg().get("servers", []))
    tasks    = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        script = _script_of(p)
        if any(pat in script for pat in patterns):
            tasks.append(p)
    return sorted(tasks, key=lambda p: p.pid)


def task_label(p) -> str:
    try:
        script = _script_of(p)
        patterns = _cfg().get("indexers", []) + _cfg().get("servers", [])
        for pat in patterns:
            if pat in script:
                parts = " ".join(p.cmdline()).split(pat, 1)
                args  = parts[1].strip()[:26] if len(parts) > 1 else ""
                return f"{pat}{(' ' + args) if args else ''}"
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return p.name()


def prio_index(p) -> int:
    try:
        v = p.nice()
        return _PRIO_VALUES.index(v) if v in _PRIO_VALUES else 2
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 2


def set_prio(p, idx: int):
    try:
        p.nice(_PRIO_VALUES[max(0, min(idx, len(_PRIO_VALUES) - 1))])
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, OSError):
        pass


def cycle_prio(p):
    set_prio(p, (prio_index(p) - 1) % len(_PRIO_VALUES))


def aff_label(p) -> str:
    try:
        n     = len(p.cpu_affinity())
        total = cpu_count()
        return "█" * n + "░" * max(0, total - n) + f" {n}/{total}"
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        return "n/a"


def cycle_affinity(p):
    total   = cpu_count()
    options = sorted({x for x in {total, max(1, total // 2), max(1, total // 4), 2, 1}
                      if x <= total})
    try:
        current = len(p.cpu_affinity())
        smaller = [x for x in options if x < current]
        p.cpu_affinity(list(range(max(smaller) if smaller else total)))
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        pass


# ── Vite helpers ──────────────────────────────────────────────────────────────

_vite_kill_confirm:   dict[str, float] = {}   # name -> timestamp of first V press
_reboot_confirm:      dict[int,  float] = {}   # pid  -> timestamp of first R press

_CREATE_NEW_CONSOLE  = 0x00000010
_DETACHED_PROCESS    = 0x00000008


def vite_probe(port: int) -> tuple[bool, int | None]:
    """Return (is_listening, pid_or_None) for a port."""
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == port and conn.status == "LISTEN":
                return True, conn.pid
    except (psutil.AccessDenied, OSError):
        pass
    return False, None


def launch_vite(app: dict) -> str:
    cwd    = str(_HERE / app.get("cwd", app["name"]))
    script = app.get("script", "dev")
    try:
        subprocess.Popen(
            f"npm run {script}",
            cwd=cwd,
            shell=True,
            creationflags=_CREATE_NEW_CONSOLE | _DETACHED_PROCESS,
        )
        return f"Launching {app['name']} on :{app['port']}…"
    except Exception as exc:
        return f"Launch failed: {exc}"


def reboot_task(p) -> str:
    """Kill p and relaunch it with the same cmdline, output → <stem>.{out,err}.log."""
    try:
        cmdline = p.cmdline()
        if not cmdline:
            return f"{RED}Cannot reboot: no cmdline{RST}"
        script  = _script_of(p)
        stem    = Path(script).stem if script else "process"
        out_log = _HERE / f"{stem}.out.log"
        err_log = _HERE / f"{stem}.err.log"
        remove_cap(p.pid)
        p.kill()
        try:
            p.wait(timeout=4)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        return f"{RED}Kill failed: {exc}{RST}"
    try:
        with open(out_log, "w") as fout, open(err_log, "w") as ferr:
            subprocess.Popen(cmdline, cwd=str(_HERE), stdout=fout, stderr=ferr)
        return f"{GRN}Rebooted {stem}{RST}"
    except Exception as exc:
        return f"{RED}Relaunch failed: {exc}{RST}"


def kill_vite(port: int) -> str:
    running, pid = vite_probe(port)
    if not running:
        return "Already stopped"
    try:
        psutil.Process(pid).terminate()
        return f"Sent SIGTERM to PID {pid}"
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        return f"Could not kill: {exc}"


# ── P-controller ──────────────────────────────────────────────────────────────

def update_controller(indexer_pids: set, indexer_cpu_sum: float, system_cpu: float) -> int:
    global _ambient_ema, _cap_ema
    n       = cpu_count()
    ambient = max(0.0, min(100.0, system_cpu - indexer_cpu_sum / n))
    with _ctrl_lock:
        _ambient_ema = 0.3 * ambient + 0.7 * _ambient_ema
        if _mode == "auto":
            desired  = max(10.0, min(90.0, 100.0 - _ambient_ema - _reserve))
            _cap_ema = 0.2 * desired + 0.8 * _cap_ema
            new_cap  = round(_cap_ema)
        else:
            new_cap  = _manual_cap
            _cap_ema = float(new_cap)
    for pid in indexer_pids:
        set_cap(pid, new_cap)
    return new_cap


# ── progress / ETA ────────────────────────────────────────────────────────────

_progress_cache: tuple = (0, 0, 0.0, "?")
_progress_cache_ts: float = 0.0
_PROGRESS_TTL = 10.0  # re-read pipeline_state at most every 10 s


def read_progress() -> tuple:
    """(indexed_count, total_count, files_per_hr, eta_str)  — cached 10 s."""
    global _progress_cache, _progress_cache_ts
    now = time.monotonic()
    if now - _progress_cache_ts < _PROGRESS_TTL:
        return _progress_cache

    indexed_count = 0
    total_count   = 0
    timestamps    = []

    try:
        raw = _PIPELINE_STATE.read_bytes()
        data = json.loads(raw.decode("utf-8", errors="replace"))
        del raw  # free immediately
        for v in data.get("files", {}).values():
            if v.get("status") == "indexed":
                indexed_count += 1
                try:
                    timestamps.append(datetime.fromisoformat(v["updated_at"]))
                except (KeyError, ValueError):
                    pass
        del data
    except (OSError, json.JSONDecodeError, AttributeError, MemoryError):
        pass

    try:
        logs = sorted(
            _LOGS_DIR.glob("*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if logs:
            txt = logs[0].read_text(encoding="utf-8", errors="replace")
            m   = re.search(r"Files to process:\s*([\d,]+)", txt)
            m2  = re.search(r"already indexed:\s*([\d,]+)", txt)
            if m:
                total_count = int(m.group(1).replace(",", ""))
                if m2:
                    total_count += int(m2.group(1).replace(",", ""))
    except (OSError, MemoryError):
        pass

    if total_count == 0:
        total_count = indexed_count

    files_per_hr = 0.0
    if len(timestamps) >= 2:
        recent   = sorted(timestamps)[-10:]
        span_hr  = (recent[-1] - recent[0]).total_seconds() / 3600
        if span_hr > 0:
            files_per_hr = (len(recent) - 1) / span_hr

    eta_str = "?"
    if files_per_hr > 0 and total_count > indexed_count:
        hrs = (total_count - indexed_count) / files_per_hr
        eta_str = (f"~{int(hrs*60)} min" if hrs < 1
                   else f"~{hrs:.1f} hr" if hrs < 48
                   else f"~{hrs/24:.1f} days")
    elif indexed_count >= total_count > 0:
        eta_str = "complete"

    result = indexed_count, total_count, files_per_hr, eta_str
    _progress_cache    = result
    _progress_cache_ts = now
    return result


# ── ANSI ──────────────────────────────────────────────────────────────────────

CLEAR = "\033[2J\033[H"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RST   = "\033[0m"
CYAN  = "\033[96m"
YEL   = "\033[93m"
RED   = "\033[91m"
GRN   = "\033[92m"
SEL   = "\033[7m"
BLU   = "\033[94m"
MAG   = "\033[95m"


def _bar(pct: float, w: int = 20) -> str:
    f = max(0, min(w, round(pct / 100 * w)))
    return "█" * f + "░" * (w - f)


def _cbar(pct: float, w: int = 20) -> str:
    color = RED if pct >= 75 else (YEL if pct >= 45 else GRN)
    return f"{color}{_bar(pct, w)}{RST}"


def _fcpu(pct: float) -> str:
    color = RED if pct >= 80 else (YEL if pct >= 40 else GRN)
    return f"{color}{pct:5.1f}%{RST}"


# ── render ────────────────────────────────────────────────────────────────────

def render(tasks: list, vite_apps: list, sel: int,
           cpu_pcts: dict, status_msg: str = "") -> str:
    W  = 76
    ts = time.strftime("%H:%M:%S")
    n_tasks = len(tasks)

    with _ctrl_lock:
        mode    = _mode
        reserve = _reserve
        man_cap = _manual_cap
        amb     = _ambient_ema
        cap     = round(_cap_ema)

    indexed, total, fph, eta = read_progress()
    cfg = _cfg()
    config_src = "  (config: throttle_config.json)" if _CONFIG_PATH.exists() else "  (config: defaults)"

    out = [
        f"{BOLD}genesis-architecture  throughput control{RST}  {DIM}{ts}{config_src}{RST}",
        f"{DIM}{'━' * W}{RST}",
    ]

    # ── controller stats ──────────────────────────────────────────────────────
    if mode == "auto":
        out += [
            f"  {BOLD}{CYAN}AUTO{RST}   "
            f"your reserve  {_cbar(reserve)}  {YEL}{reserve:>3}%{RST}"
            f"  {DIM}←/→   M manual{RST}",
            f"         ambient load  {_cbar(amb)}  {amb:>4.0f}%  {DIM}(2 s EMA){RST}",
            f"         indexer cap   {_cbar(cap)}  {MAG}{cap:>3}%{RST}"
            + (f"  {DIM}{indexed:,}/{total:,} files{RST}  ETA {CYAN}{eta}{RST}"
               if fph > 0 else f"  {DIM}{indexed:,} files indexed{RST}"),
        ]
    else:
        auto_cap = max(10, min(90, round(100 - amb - reserve)))
        delta    = auto_cap - man_cap
        hint     = (f"  {DIM}→ auto {auto_cap}% ({delta:+d} pp){RST}"
                    if abs(delta) > 3 else "")
        out += [
            f"  {BOLD}{YEL}MANUAL{RST} "
            f"indexer cap   {_cbar(man_cap)}  {YEL}{man_cap:>3}%{RST}"
            f"  {DIM}←/→   M auto{RST}{hint}",
            f"         ambient load  {_cbar(amb)}  {amb:>4.0f}%  {DIM}(2 s EMA){RST}",
            f"         "
            + (f"  {DIM}{indexed:,}/{total:,} files{RST}  ETA {CYAN}{eta}{RST}"
               if fph > 0 else f"  {DIM}{indexed:,} files indexed{RST}"),
        ]

    # ── pipeline tasks ────────────────────────────────────────────────────────
    out.append(f"{DIM}{'━' * W}{RST}")
    out.append(f"  {'PIPELINE':<34} {'PID':>6}  {'CPU%':>6}  {'MEM':>6}  CORES")
    out.append(f"{DIM}{'─' * W}{RST}")

    if not tasks:
        out.append(f"  {DIM}(no pipeline processes running){RST}")
    else:
        for i, p in enumerate(tasks):
            si = i
            try:
                label  = task_label(p)[:33]
                pct    = cpu_pcts.get(p.pid, 0.0)
                mem_mb = p.memory_info().rss / 1048576
                aff    = aff_label(p)
                tag    = f"{DIM}[idx]{RST}" if is_indexer(p) else f"{DIM}[srv]{RST}"

                cols = (f"{p.pid:>6}  {_fcpu(pct)}  "
                        f"{mem_mb:>5.0f}M  {BLU}{aff}{RST}")

                row = f"  {label:<33} {tag} {cols}"
                out.append(f"{SEL}{row}{RST}" if si == sel else row)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                out.append(f"  {DIM}(process gone){RST}")

    # ── Vite apps ─────────────────────────────────────────────────────────────
    out.append(f"{DIM}{'─' * W}{RST}")
    out.append(f"  {'VITE APPS':<34}  {'PORT':>5}  STATUS")
    out.append(f"{DIM}{'─' * W}{RST}")

    if not vite_apps:
        out.append(f"  {DIM}(none configured in throttle_config.json){RST}")
    else:
        for j, app in enumerate(vite_apps):
            si      = n_tasks + j
            running, pid = vite_probe(app["port"])
            confirm = app["name"] in _vite_kill_confirm and \
                      time.time() - _vite_kill_confirm[app["name"]] < 3.0

            if running:
                dot    = f"{GRN}●{RST}"
                status = f"{GRN}RUNNING{RST}  PID {pid}"
                if confirm:
                    status += f"  {YEL}press V again to stop{RST}"
            else:
                dot    = f"{DIM}○{RST}"
                status = f"{DIM}stopped{RST}  press V to launch"

            row = f"  {dot} {app['name']:<14} :{app['port']}  {status}"
            out.append(f"{SEL}{row}{RST}" if si == sel else row)

    # ── help bar ──────────────────────────────────────────────────────────────
    out.append(f"{DIM}{'─' * W}{RST}")
    try:
        all_count = n_tasks + len(vite_apps)
        if all_count > 0 and 0 <= sel < all_count:
            if sel < n_tasks:
                # pipeline task selected
                pi   = prio_index(tasks[sel])
                plbl = _PRIO_LABELS.get(_PRIO_VALUES[pi], "?")
                out.append(
                    f"  {CYAN}A{RST} affinity  "
                    f"{CYAN}P{RST} priority [{DIM}{plbl}{RST}]  "
                    f"{YEL}R R{RST} reboot  "
                    f"{RED}K K{RST} kill  "
                    f"{DIM}Q quit{RST}"
                )
            else:
                # Vite app selected
                out.append(
                    f"  {CYAN}V{RST} launch / stop  "
                    f"{DIM}Q quit{RST}"
                )
        else:
            out.append(f"  {DIM}Q quit{RST}")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        out.append(f"  {DIM}Q quit{RST}")

    if status_msg:
        out.append(f"  {status_msg}")

    return "\n".join(out)


# ── console + main loop ───────────────────────────────────────────────────────

def _setup_console():
    try:
        if sys.platform == "win32":
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleOutputCP(65001)
            k.SetConsoleCP(65001)
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
    except Exception:
        pass


def run():
    global _mode, _reserve, _manual_cap

    _setup_console()

    sel               = 0
    tasks             = []
    cpu_pcts:         dict[int, float] = {}
    last_refresh      = 0.0
    status_msg        = ""
    kill_pending_pid  = None
    kill_pending_time = 0.0

    # Prime cpu_percent
    psutil.cpu_percent(interval=None)

    print(CLEAR, end="", flush=True)

    while True:
        now = time.time()

        # expire kill confirm
        if kill_pending_pid is not None and now - kill_pending_time > 3.0:
            kill_pending_pid = None
            status_msg       = ""
            last_refresh     = 0

        # expire vite + reboot confirms
        for name in list(_vite_kill_confirm):
            if now - _vite_kill_confirm[name] > 3.0:
                del _vite_kill_confirm[name]
                last_refresh = 0
        for pid2 in list(_reboot_confirm):
            if now - _reboot_confirm[pid2] > 3.0:
                del _reboot_confirm[pid2]
                last_refresh = 0

        if now - last_refresh >= 2.0:
            tasks      = get_tasks()
            vite_apps  = _cfg().get("vite_apps", _DEFAULT_CONFIG["vite_apps"])
            total_items = len(tasks) + len(vite_apps)
            sel        = min(sel, max(0, total_items - 1))

            # CPU sampling
            system_cpu = psutil.cpu_percent(interval=None)
            for p in tasks:
                try:
                    p.cpu_percent(interval=None)   # prime on first cycle
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Remove caps for dead processes
            live_pids = {p.pid for p in tasks}
            for pid in list(_cap_threads):
                if pid not in live_pids:
                    remove_cap(pid)

            cpu_pcts = {}
            for p in tasks:
                try:
                    cpu_pcts[p.pid] = p.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    cpu_pcts[p.pid] = 0.0

            indexer_pids    = {p.pid for p in tasks if is_indexer(p)}
            indexer_cpu_sum = sum(cpu_pcts.get(pid, 0.0) for pid in indexer_pids)
            update_controller(indexer_pids, indexer_cpu_sum, system_cpu)

            last_refresh = now
            print(CLEAR + render(tasks, vite_apps, sel, cpu_pcts, status_msg),
                  end="", flush=True)

        # ── keyboard ──────────────────────────────────────────────────────────
        if msvcrt.kbhit():
            ch = msvcrt.getch()

            if ch in (b"q", b"Q"):
                print(CLEAR, end="")
                break

            elif ch in (b"m", b"M"):
                with _ctrl_lock:
                    _mode = "manual" if _mode == "auto" else "auto"
                status_msg       = (f"{CYAN}AUTO — adaptive control{RST}"
                                    if _mode == "auto"
                                    else f"{YEL}MANUAL — cap fixed at {_manual_cap}%{RST}")
                kill_pending_pid = None
                last_refresh     = 0

            elif ch in (b"\xe0", b"\x00"):   # extended key
                ch2 = msvcrt.getch()
                kill_pending_pid = None

                if ch2 == b"H":    # ↑
                    sel        = max(0, sel - 1)
                    status_msg = ""
                elif ch2 == b"P":  # ↓
                    vite_apps  = _cfg().get("vite_apps", [])
                    sel        = min(len(tasks) + len(vite_apps) - 1, sel + 1)
                    status_msg = ""
                elif ch2 in (b"K", b"M"):   # ←/→
                    delta = -10 if ch2 == b"K" else 10
                    with _ctrl_lock:
                        if _mode == "auto":
                            _reserve   = max(10, min(90, _reserve + delta))
                            status_msg = f"{CYAN}Reserve → {_reserve}%{RST}"
                        else:
                            _manual_cap = max(10, min(90, _manual_cap + delta))
                            status_msg  = f"{YEL}Cap → {_manual_cap}%{RST}"

                last_refresh = 0

            # ── pipeline-task actions ─────────────────────────────────────────
            elif sel < len(tasks):
                target = tasks[sel]

                if ch in (b"a", b"A"):
                    kill_pending_pid = None; status_msg = ""
                    cycle_affinity(target)
                    last_refresh = 0

                elif ch in (b"p", b"P"):
                    kill_pending_pid = None; status_msg = ""
                    cycle_prio(target)
                    last_refresh = 0

                elif ch in (b"r", b"R"):
                    pid2 = target.pid
                    if pid2 in _reboot_confirm and now - _reboot_confirm[pid2] < 3.0:
                        del _reboot_confirm[pid2]
                        kill_pending_pid = None
                        status_msg = reboot_task(target)
                    else:
                        _reboot_confirm[pid2] = now
                        label = task_label(target)
                        status_msg = (f"{YEL}R again within 3 s to reboot "
                                      f"{label} (PID {pid2}){RST}")
                    last_refresh = 0

                elif ch in (b"k", b"K"):
                    if kill_pending_pid == target.pid:
                        try:
                            label = task_label(target)
                            pid   = target.pid
                            remove_cap(pid)
                            target.terminate()
                            status_msg = f"{RED}SIGTERM → {label} (PID {pid}){RST}"
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            status_msg = f"{YEL}Already gone{RST}"
                        kill_pending_pid = None
                    else:
                        label            = task_label(target)
                        kill_pending_pid  = target.pid
                        kill_pending_time = now
                        status_msg = f"{YEL}K again within 3 s to kill {label} (PID {target.pid}){RST}"
                    last_refresh = 0

                else:
                    kill_pending_pid = None; status_msg = ""

            # ── Vite-app actions ──────────────────────────────────────────────
            else:
                vite_apps = _cfg().get("vite_apps", [])
                vi        = sel - len(tasks)
                if 0 <= vi < len(vite_apps):
                    app  = vite_apps[vi]
                    name = app["name"]

                    if ch in (b"v", b"V"):
                        running, pid = vite_probe(app["port"])
                        if running:
                            if name in _vite_kill_confirm:
                                # second V — kill
                                del _vite_kill_confirm[name]
                                status_msg = kill_vite(app["port"])
                            else:
                                _vite_kill_confirm[name] = now
                                status_msg = f"{YEL}V again within 3 s to stop {name}{RST}"
                        else:
                            _vite_kill_confirm.pop(name, None)
                            status_msg = launch_vite(app)
                        last_refresh = 0
                    else:
                        _vite_kill_confirm.pop(name, None)
                        status_msg = ""

        time.sleep(0.05)


if __name__ == "__main__":
    run()
