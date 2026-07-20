"""EngineClient — drives the engine child process from the GUI.

Spawns ``python -m studio.engine_process`` and communicates over a
multiprocessing.connection socket. A tiny reader QThread blocks on recv()
(which releases the GIL) and re-emits results as Qt signals on the UI thread.
All heavy CUDA work happens in the child process, so the GUI never freezes and
never hits the thread+CUDA hang.

The Qt signal interface matches the old in-process Worker, so the window code
is unchanged apart from construction.
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
import tempfile
import threading
from multiprocessing.connection import Listener

from PySide6.QtCore import QObject, QThread, Signal

from .engine import REPO_ROOT


class _Reader(QThread):
    """Blocks on conn.recv() and forwards each (event, payload) to the UI."""

    event = Signal(str, object)

    def __init__(self, conn) -> None:
        super().__init__()
        self._conn = conn
        self._alive = True

    def run(self) -> None:
        reason = None
        while self._alive:
            try:
                msg = self._conn.recv()
            except (EOFError, OSError):
                break                    # pipe closed: child exited, or we tore it down
            except Exception as exc:     # noqa: BLE001
                # A frame that won't unpickle. recv() raises UnpicklingError /
                # AttributeError / ImportError here — none of them OSError — so
                # uncaught it kills this thread silently and the GUI waits forever
                # for a reply that can never arrive. Report it and stand down.
                reason = (f"The engine sent a message this app couldn't read ({exc}). "
                          "Restart the app to continue.")
                break
            if isinstance(msg, tuple) and msg:   # ignore malformed frames, don't die
                self.event.emit(msg[0], msg[1] if len(msg) > 1 else None)
        if self._alive:   # broke out unexpectedly (child crashed) — not a clean stop()
            # "died", not "error": an engine-side error means the child caught an
            # exception and is still alive and usable, but this means the child is
            # GONE and the client has to reap it.
            self.event.emit("died", reason or (
                "The engine process stopped unexpectedly (it may have run out of GPU "
                "memory or crashed). Restart the app to continue."
            ))

    def stop(self) -> None:
        self._alive = False


class EngineClient(QObject):
    progress = Signal(str)
    modelLoaded = Signal(str)
    inferenceDone = Signal(object)
    cloudDone = Signal(object)
    error = Signal(str)
    busy = Signal(bool)
    vram = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self._proc = None
        self._conn = None
        self._reader = None
        self._logf = None
        self._stopped = False
        self._send_lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._spawn(None)

    def _spawn(self, python_exe: str | None = None) -> None:
        """Spawn a fresh engine child (optionally with a per-model interpreter)
        and wire its reader. Reassigns self._proc/_conn/_reader/_logf."""
        exe = python_exe or sys.executable
        authkey = secrets.token_bytes(24)
        listener = Listener(("127.0.0.1", 0), authkey=authkey)
        port = listener.address[1]
        # Bound accept(): a child that dies before connecting (import error, bad
        # env, OOM at startup) would otherwise hang the GUI thread forever.
        try:
            listener._listener._socket.settimeout(30)
        except Exception:
            pass
        # Give the child real stdout/stderr (pythonw would otherwise hand it
        # None, crashing torch.hub) and capture its output for debugging.
        self._logf = open(
            os.path.join(tempfile.gettempdir(), "fs_studio_engine.log"),
            "w", encoding="utf-8", errors="replace",
        )
        try:
            self._proc = subprocess.Popen(
                [exe, "-m", "studio.engine_process", str(port), authkey.hex()],
                cwd=REPO_ROOT,
                env=dict(os.environ),
                stdout=self._logf,
                stderr=subprocess.STDOUT,
            )
            self._conn = listener.accept()   # child connects before it imports torch
        except Exception as exc:   # startup/connect failed — don't leak fd/socket/proc
            try: listener.close()
            except Exception: pass
            if self._proc is not None:
                try: self._proc.kill()
                except Exception: pass
            if self._logf is not None:
                try: self._logf.close()
                except Exception: pass
                self._logf = None
            raise RuntimeError(
                "The engine process failed to start — see the log at "
                f"{os.path.join(tempfile.gettempdir(), 'fs_studio_engine.log')}"
            ) from exc
        listener.close()
        self._reader = _Reader(self._conn)
        self._reader.event.connect(self._dispatch)
        self._reader.start()

    def _teardown_child(self) -> None:
        """Stop the reader, close the pipe, reap the child. Does NOT set the
        permanent _stopped flag — reused by both stop() and switchBackend()."""
        # stop() the reader BEFORE asking the child to quit. The child can exit and
        # drop the socket before the next line runs, and a reader still marked alive
        # reads that clean EOF as a crash — firing "the engine stopped unexpectedly"
        # into a switch the user asked for, which mid-sweep banks a healthy model as
        # failed.
        if self._reader is not None:
            self._reader.stop()
        try:                   # best-effort graceful quit; the child also exits on EOF
            if self._conn is not None:
                with self._send_lock:
                    self._conn.send(("quit",))
        except Exception:
            pass
        if self._conn is not None:
            try:
                self._conn.close()   # unblocks the reader's blocked recv()
            except Exception:
                pass
        if self._reader is not None:
            self._reader.wait(2000)  # join before destruction (no "QThread still running")
        if self._proc is not None:
            try:
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=2)   # reap the killed child
                except Exception:
                    pass
        if self._logf is not None:
            try:
                self._logf.close()
            except Exception:
                pass
        self._reader = self._conn = self._proc = self._logf = None

    @property
    def alive(self) -> bool:
        """Is the engine child up and connected? False after a 'died' teardown, so a
        batch can tell a one-off pair failure (carry on) from a dead engine (stop)."""
        return self._conn is not None and self._proc is not None

    def stop(self) -> None:
        if self._stopped:   # idempotent: closeEvent AND app.aboutToQuit both call this
            return
        self._stopped = True   # from here _send() is a silent no-op (no shutdown dialogs)
        self._teardown_child()

    def switchBackend(self, python_exe: str | None, backend_key: str, ckpt: str) -> None:
        """Replace the engine child with a fresh one (its own interpreter, for
        per-model venvs) and load the new model. Used when the user picks a
        different model or checkpoint — a fresh process guarantees clean VRAM."""
        if self._stopped:
            return
        self.busy.emit(True)
        self._teardown_child()
        try:
            self._spawn(python_exe)
        except Exception as exc:  # noqa: BLE001
            self.busy.emit(False)
            self.error.emit(f"Couldn't start the engine for this model: {exc}")
            return
        self.loadModel(backend_key, ckpt)

    # --------------------------------------------------------------- inbound
    def _dispatch(self, ev: str, data) -> None:
        if ev == "progress":
            self.progress.emit(data)
        elif ev == "loaded":
            self.busy.emit(False)
            self.modelLoaded.emit(data)
        elif ev == "inference":
            self.inferenceDone.emit(data)
        elif ev == "cloud":
            self.cloudDone.emit(data)
            self.busy.emit(False)
        elif ev == "error":
            self.error.emit(data)
            self.busy.emit(False)
        elif ev == "died":
            # Reap the corpse NOW. Leaving _conn pointing at a dead socket means the
            # 1.5s VRAM poll keeps raising into another modal error dialog, and each
            # of those runs a nested event loop that lets the next poll fire — an
            # unkillable stack of dialogs. After teardown _conn is None and the poll
            # is a silent no-op; switchBackend() can still spawn a fresh child, so
            # picking another model recovers the app.
            self._teardown_child()
            self.busy.emit(False)   # clear busy first: error.emit() blocks on a modal
            self.error.emit(data)
        elif ev == "vram":
            self.vram.emit(float(data[0]), float(data[1]))

    # -------------------------------------------------------------- outbound
    def _send(self, msg, quiet: bool = False) -> None:
        if self._stopped:
            return   # shutting down — the pipe is (being) closed; never send/alert again
        try:
            if self._conn is None:
                # Not merely "nothing to do": every caller has already emitted
                # busy(True), so returning quietly latches the UI busy forever with
                # no error and no reply. Fail loudly instead.
                raise RuntimeError("the engine process is not running")
            with self._send_lock:
                self._conn.send(msg)
        except Exception as exc:  # noqa: BLE001
            if self._stopped:
                return   # a close raced this send — expected during shutdown, stay silent
            self.busy.emit(False)   # don't leave the UI stuck "busy" on a dead pipe
            if quiet:
                return   # background poll: report by not answering, never by nagging
            self.error.emit(f"Lost connection to engine process: {exc}")

    def loadModel(self, backend_key: str, ckpt: str) -> None:
        self.busy.emit(True)
        self._send(("load", backend_key, ckpt))

    def runInference(self, left, right, params) -> None:
        self.busy.emit(True)
        self._send(("infer", left, right, params))

    def rebuildCloud(self, result, params) -> None:
        """Rebuild the cloud from the child's CACHED result (cheap — only params
        cross the socket). Pass `result` to rebuild from THAT one instead: after a
        comparison the GUI can be showing a model this child never ran, so its
        cache would rebuild the wrong model's cloud."""
        self.busy.emit(True)   # so the UI can throttle/coalesce during a rebuild
        self._send(("rebuild", params, result))

    def rescaleResult(self, factor: float) -> None:
        """Rescale the child's CACHED result (depth + baseline) by ``factor`` after
        a mm⇄m unit switch, so subsequent live cloud rebuilds — which the child
        builds from that cached result — come back in the new unit. Cheap (one
        float over the socket) and fire-and-forget (no reply, doesn't touch busy)."""
        self._send(("rescale", float(factor)))

    def requestVram(self) -> None:
        # quiet: this is the only sender on a timer, so it's the only one that can
        # turn a dead pipe into a repeating dialog. A stale VRAM readout is a far
        # better failure than a modal the user can't out-click.
        self._send(("vram",), quiet=True)
