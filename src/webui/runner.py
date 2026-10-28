"""子进程管线运行器：真停止 + 日志落盘 + 单 worker 队列。

为什么用子进程而非 in-process：
- config 是全局单例，webui 长驻进程里反复 load/override 会互相污染；
- 后台线程跑管线无法真正终止（daemon 线程 kill 不掉），子进程可 killpg；
- ASR 模型本就在 transcribe() 内加载用完即释放，子进程无额外冷加载代价。
"""
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "src" / "main.py"
LOG_DIR = REPO_ROOT / "logs" / "webui"

_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")


@dataclass
class Job:
    job_id: int
    video_name: str
    overrides: dict
    force_steps: list  # step key 列表
    skip_steps: list
    status: str = "queued"       # queued | running | succeeded | failed | stopped
    log: str = ""
    log_path: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    _proc: object = field(default=None, repr=False)


class JobRunner:
    def __init__(self, saved_proxy: dict, secrets: list = None):
        self._saved_proxy = {k: v for k, v in (saved_proxy or {}).items() if v}
        self._secrets = [s for s in (secrets or []) if s and len(str(s)) >= 6]
        self._jobs = {}
        self._queue = []
        self._current = None
        self._seq = 0
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 提交 / 查询 ────────────────────────────────────────────────
    def submit(self, video_name, overrides, force_steps, skip_steps) -> int:
        with self._lock:
            self._seq += 1
            job = Job(self._seq, video_name, overrides or {},
                      list(force_steps or []), list(skip_steps or []))
            self._jobs[job.job_id] = job
            self._queue.append(job.job_id)
        return job.job_id

    def get(self, job_id) -> Job:
        return self._jobs.get(job_id)

    def queue_view(self) -> str:
        with self._lock:
            rows = []
            if self._current is not None:
                j = self._jobs[self._current]
                rows.append(f"▶ #{j.job_id} EP{j.video_name} 运行中")
            for jid in self._queue:
                rows.append(f"⋯ #{jid} EP{self._jobs[jid].video_name} 排队")
            recent = sorted((j for j in self._jobs.values()
                             if j.status in ("succeeded", "failed", "stopped")),
                            key=lambda j: j.ended_at, reverse=True)[:6]
            icon = {"succeeded": "✅", "failed": "❌", "stopped": "⏹"}
            for j in recent:
                dur = f"{j.ended_at - j.started_at:.0f}s" if j.started_at else ""
                rows.append(f"{icon[j.status]} #{j.job_id} EP{j.video_name} {j.status} {dur}")
        return "\n".join(rows) if rows else "（空闲）"

    def is_active(self, job_id) -> bool:
        j = self._jobs.get(job_id)
        return bool(j) and j.status in ("queued", "running")

    # ── 停止 ──────────────────────────────────────────────────────
    def stop(self, job_id=None):
        """停止指定 job（默认当前运行的）：从队列移除排队中的；正在跑的 killpg。"""
        with self._lock:
            target = job_id if job_id is not None else self._current
            # 排队中的直接移除
            if target in self._queue:
                self._queue.remove(target)
                self._jobs[target].status = "stopped"
                self._jobs[target].ended_at = time.time()
                return
            job = self._jobs.get(target)
            proc = job._proc if job else None
        if proc and proc.poll() is None:
            self._kill_proc_tree(proc)
            if job:
                job.status = "stopped"

    def stop_all(self):
        """停止全部：清空整个排队 + 杀掉当前运行的。用于批量任务的「停止全部」。"""
        with self._lock:
            for jid in self._queue:
                self._jobs[jid].status = "stopped"
                self._jobs[jid].ended_at = time.time()
            self._queue.clear()
        self.stop()  # 杀当前运行的

    def _kill_proc_tree(self, proc):
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
        # 先尝试 psutil 收集孙进程（ffmpeg），再整组 SIGTERM→SIGKILL
        try:
            import psutil
            children = psutil.Process(proc.pid).children(recursive=True)
        except Exception:
            children = []
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(25):  # 最多等 5s
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        if proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for c in children:
            try:
                c.kill()
            except Exception:
                pass

    # ── 日志脱敏 ──────────────────────────────────────────────────
    def _redact(self, line: str) -> str:
        for s in self._secrets:
            if s in line:
                line = line.replace(s, s[:3] + "****" + s[-2:])
        return line

    # ── worker ────────────────────────────────────────────────────
    def _loop(self):
        while True:
            with self._lock:
                if self._queue:
                    jid = self._queue.pop(0)
                    self._current = jid
                else:
                    jid = None
            if jid is None:
                time.sleep(0.2)
                continue
            self._run_job(self._jobs[jid])
            with self._lock:
                self._current = None

    def _build_cmd(self, job: Job, overrides_path: str):
        cmd = ["python", "-u", str(MAIN_PY), "--video", job.video_name,
               "--overrides", overrides_path]
        for k in job.force_steps:
            cmd.append(f"--force-{k.replace('_', '-')}")
        for k in job.skip_steps:
            cmd.append(f"--skip-{k.replace('_', '-')}")
        return cmd

    def _run_job(self, job: Job):
        if job.status == "stopped":  # 排队时已被取消
            return
        job.status = "running"
        job.started_at = time.time()
        ts = time.strftime("%Y%m%d_%H%M%S")
        job.log_path = str(LOG_DIR / f"{ts}_ep{job.video_name}.log")

        ov = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(job.overrides, ov, ensure_ascii=False)
        ov.close()

        env = dict(os.environ)
        env.update(self._saved_proxy)   # 管线内 LLM/edge-tts 需要代理
        env["PYTHONUNBUFFERED"] = "1"

        cmd = self._build_cmd(job, ov.name)
        job.log += f"$ {' '.join(cmd)}\n" + "─" * 50 + "\n"

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True)
        except Exception as e:
            job.log += f"[ERROR] 启动子进程失败: {e}\n"
            job.status = "failed"
            job.ended_at = time.time()
            return
        job._proc = proc

        with open(job.log_path, "w", encoding="utf-8") as lf:
            for raw in proc.stdout:
                line = self._redact(raw)
                job.log += line
                lf.write(line)
                lf.flush()
        proc.wait()
        job.ended_at = time.time()
        Path(ov.name).unlink(missing_ok=True)

        if job.status == "stopped":
            job.log += "\n" + "─" * 50 + "\n⏹ 已停止\n"
        elif proc.returncode == 0:
            job.status = "succeeded"
            job.log += "\n" + "─" * 50 + "\n✅ 完成\n"
        else:
            job.status = "failed"
            job.log += f"\n{'─' * 50}\n❌ 失败 (exit {proc.returncode})\n"


def list_history_logs(limit: int = 40):
    if not LOG_DIR.exists():
        return []
    logs = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in logs[:limit]]


def read_history_log(name: str) -> str:
    if not name:
        return ""
    p = LOG_DIR / Path(name).name  # 防目录穿越
    if p.exists() and p.parent == LOG_DIR:
        return p.read_text(encoding="utf-8", errors="replace")
    return "(日志不存在)"
