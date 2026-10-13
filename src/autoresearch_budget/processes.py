"""Host subprocess ownership and CLI cancellation on POSIX platforms."""
from contextlib import contextmanager
import os
import signal
import subprocess
from threading import Event, RLock, current_thread, main_thread

_ACTIVE = set()
_LOCK = RLock()
_CANCELLED = Event()


def _signal(process, sig):
    try:
        if os.name == 'posix':
            os.killpg(process.pid, sig)
        elif process.poll() is None:
            process.terminate() if sig == signal.SIGTERM else process.kill()
    except ProcessLookupError:
        pass


def cancel_all():
    _CANCELLED.set()
    with _LOCK:
        active=list(_ACTIVE)
    for process in active:_signal(process,signal.SIGTERM)


def run_command(command, *, check=False, timeout=None, **kwargs):
    process=subprocess.Popen(command,start_new_session=os.name=='posix',**kwargs)
    with _LOCK:_ACTIVE.add(process)
    try:
        if _CANCELLED.is_set():
            raise InterruptedError("host execution cancelled")
        stdout,stderr=process.communicate(timeout=timeout)
        result=subprocess.CompletedProcess(command,process.returncode,stdout,stderr)
        if check:result.check_returncode()
        return result
    finally:
        # Retain no child background processes after this owned attempt ends.
        _signal(process,signal.SIGTERM)
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:_signal(process,signal.SIGKILL);process.wait()
        # The parent can finish before a surviving descendant; target the group.
        _signal(process,signal.SIGKILL)
        with _LOCK:_ACTIVE.discard(process)


@contextmanager
def cancellation_scope():
    previous={}
    if current_thread() is main_thread():_CANCELLED.clear()
    def interrupt(signum, frame):
        cancel_all()
        raise KeyboardInterrupt
    if current_thread() is main_thread():
        for sig in (signal.SIGINT,signal.SIGTERM):
            previous[sig]=signal.signal(sig,interrupt)
    try:yield
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
        if previous:_CANCELLED.clear()
