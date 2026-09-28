"""Pinned OpenAI HumanEval execution core (MIT).

Adapted from ``human_eval/execution.py`` at commit
6d43fb980f9fee3c892a914eda09951f772ad10d. The project keeps the official
prompt+completion+test construction and process/time-limit/reliability behavior.
"""

from __future__ import annotations

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile


def unsafe_execute(problem, completion, timeout, result):
    with create_tempdir():
        import os
        import shutil

        rmtree, rmdir, chdir_fn = shutil.rmtree, os.rmdir, os.chdir
        reliability_guard()
        check_program = (
            problem["prompt"] + completion + "\n" + problem["test"]
            + "\n" + f"check({problem['entry_point']})"
        )
        try:
            with swallow_io(), time_limit(timeout):
                exec(check_program, {})
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as error:
            result.append(f"failed: {error}")
        shutil.rmtree, os.rmdir, os.chdir = rmtree, rmdir, chdir_fn


def check_correctness(problem, completion, timeout, completion_id=None):
    manager = multiprocessing.Manager()
    result = manager.list()
    process = multiprocessing.Process(
        target=unsafe_execute, args=(problem, completion, timeout, result)
    )
    process.start()
    process.join(timeout=timeout + 1)
    if process.is_alive():
        process.kill()
    if not result:
        result.append("timed out")
    return {
        "task_id": problem["task_id"], "passed": result[0] == "passed",
        "result": result[0], "completion_id": completion_id,
    }


@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(_signum, _frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), redirect_stdin(stream):
        yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname, chdir(dirname):
        yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *_args, **_kwargs):
        raise OSError

    readline = read
    readlines = read

    def readable(self, *_args, **_kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes=None):
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
    faulthandler.disable()
    import builtins
    import shutil
    import subprocess

    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for name in (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
        "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
        "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
        "lchflags", "lchmod", "lchown", "getcwd", "chdir",
    ):
        if hasattr(os, name):
            setattr(os, name, None)
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None
    builtins.help = None
    import sys
    for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[name] = None
