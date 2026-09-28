"""Minimal LiveCodeBench code-generation test runner.

Adapted from ``lcb_runner/evaluation/testing_util.py`` at commit
28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24, licensed MIT. Local changes remove
debug output and unrelated dependencies while retaining call-based/stdin test
semantics, imports, normalization, timeout codes, and the reliability guard.
"""

from __future__ import annotations

import ast
import faulthandler
import json
import signal
import sys
import time
from decimal import Decimal
from io import StringIO
from types import ModuleType
from unittest.mock import mock_open, patch


IMPORT_STRING = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string, re, datetime, collections, heapq, bisect, copy, math, random
import statistics, itertools, functools, operator, io, sys, json
sys.setrecursionlimit(50000)
"""


class TimeoutException(Exception):
    pass


def _timeout_handler(_signum, _frame):
    raise TimeoutException("timeout")


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda *_args: None
        return self

    def __exit__(self, *_args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")

    def read(self, *_args):
        return self.inputs

    def readline(self, *_args):
        return self.inputs.split(b"\n")[0] + b"\n"


class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *_args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


def clean_if_name(code: str) -> str:
    try:
        tree = ast.parse(code)
        last = tree.body[-1]
        if isinstance(last, ast.If) and ast.unparse(last.test).strip() == "__name__ == '__main__'":
            return ast.unparse(tree.body[:-1]) + "\n" + ast.unparse(last.body)
    except Exception:
        pass
    return code


def make_function(code: str) -> str:
    try:
        tree = ast.parse(code)
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        body = [node for node in tree.body if node not in imports]
        function = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=body,
            decorator_list=[],
            lineno=-1,
        )
        return IMPORT_STRING + "\n" + ast.unparse(imports) + "\n" + ast.unparse(function)
    except Exception:
        return code


def _call_method(method, inputs: str):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    iterator = iter(inputs.split("\n"))
    stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", stdin)
    @patch("sys.stdin.readline", lambda *_args: next(iterator))
    @patch("sys.stdin.readlines", lambda *_args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *_args: inputs)
    def inner(current):
        try:
            return current()
        except SystemExit:
            return None

    return inner(method)


def _compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        module = ModuleType("tmp_sol", "")
        exec(code, module.__dict__)
        return module.Solution() if "class Solution" in code else module
    finally:
        signal.alarm(0)


def _truncate(value, length=300):
    value = value if isinstance(value, str) else str(value)
    if len(value) <= length:
        return value
    return value[: length // 2] + "...(truncated) ..." + value[-length // 2 :]


def _grade_call_based(code, inputs, outputs, fn_name, timeout):
    compiled = _compile_code(IMPORT_STRING + "\n\n" + code, timeout)
    method = getattr(compiled, fn_name, None)
    if method is None:
        return [-4], {"error_code": -4, "error_message": "Runtime Error: function missing"}
    decoded_inputs = [[json.loads(line) for line in item.split("\n")] for item in inputs]
    decoded_outputs = [json.loads(item) for item in outputs]
    results = []
    total = 0.0
    for arguments, expected in zip(decoded_inputs, decoded_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            started = time.time()
            prediction = method(*arguments)
            total += time.time() - started
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            if prediction != expected:
                results.append(-2)
                return results, {
                    "output": _truncate(prediction), "inputs": _truncate(arguments),
                    "expected": _truncate(expected), "error_code": -2,
                    "error_message": "Wrong Answer",
                }
            results.append(True)
        except Exception as error:
            code_value = -3 if isinstance(error, TimeoutException) else -4
            results.append(code_value)
            return results, {
                "error": repr(error), "error_code": code_value,
                "error_message": "Time Limit Exceeded" if code_value == -3 else "Runtime Error",
                "inputs": _truncate(arguments), "expected": _truncate(expected),
            }
        finally:
            signal.alarm(0)
            faulthandler.disable()
    return results, {"execution time": total}


def _decimal_line(line: str):
    try:
        return [Decimal(item) for item in line.split()]
    except Exception:
        return None


def _stripped_lines(value: str):
    return [line.strip() for line in value.strip().split("\n")]


def _grade_stdio(code, inputs, outputs, timeout):
    compiled = _compile_code(make_function(clean_if_name(code)), timeout)
    method = getattr(compiled, "wrapped_function", None)
    if method is None:
        return [-4], {"error_code": -4, "error_message": "Runtime Error: main missing"}
    results = []
    total = 0.0
    for stdin, expected in zip(inputs, outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        with Capturing() as captured:
            try:
                started = time.time()
                _call_method(method, stdin)
                total += time.time() - started
            except Exception as error:
                code_value = -3 if isinstance(error, TimeoutException) else -4
                results.append(code_value)
                return results, {
                    "error": repr(error), "error_code": code_value,
                    "error_message": "Time Limit Exceeded" if code_value == -3 else "Runtime Error",
                }
            finally:
                signal.alarm(0)
                faulthandler.disable()
        predicted_lines = _stripped_lines(captured[0])
        expected_lines = _stripped_lines(expected)
        if len(predicted_lines) != len(expected_lines):
            return results + [-2], {"error_code": -2, "error_message": "Wrong answer: mismatched output length"}
        for predicted, wanted in zip(predicted_lines, expected_lines):
            if predicted == wanted:
                continue
            predicted_decimal = _decimal_line(predicted)
            wanted_decimal = _decimal_line(wanted)
            if predicted_decimal is None or predicted_decimal != wanted_decimal:
                return results + [-2], {"error_code": -2, "error_message": "Wrong Answer"}
        results.append(True)
    return results, {"execution time": total}


def run_test(sample, test=None, debug=False, timeout=6):
    del debug
    signal.signal(signal.SIGALRM, _timeout_handler)
    reliability_guard()
    payload = json.loads(sample["input_output"])
    if test is None:
        raise AssertionError("test code is required")
    timeout = max(1, int(timeout))
    if payload.get("fn_name") is None:
        return _grade_stdio(test, payload["inputs"], payload["outputs"], timeout)
    return _grade_call_based(
        test, payload["inputs"], payload["outputs"], payload["fn_name"], timeout
    )


def reliability_guard(maximum_memory_bytes=None):
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
    faulthandler.disable()
    import builtins
    import os
    import shutil
    import subprocess

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
    for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[name] = None
