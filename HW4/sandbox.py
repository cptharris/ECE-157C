"""
sandbox.py

Maintain a single namespace = {} dict that persists across calls.
Expose one function: execute(code: str) -> dict — runs exec(code, namespace), captures stdout/stderr via io.StringIO redirect, returns {"stdout": ..., "stderr": ..., "error": None | str}.
Keep it completely stateless as a module — the caller owns the namespace lifetime.
"""

from typing import TypedDict, Optional


class ExecutionResult(TypedDict):
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


import sys
from io import StringIO
import traceback

namespace = {}


def execute(code: str) -> ExecutionResult:
    global namespace
    result = None
    capture_out_stream = StringIO()
    capture_err_stream = StringIO()

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = capture_out_stream
    sys.stderr = capture_err_stream

    try:
        exec(code, namespace)
        result = ExecutionResult(
            stdout=capture_out_stream.getvalue(), stderr=capture_err_stream.getvalue()
        )
    except Exception as e:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        traceback.print_exc()
        result = ExecutionResult(
            stdout=capture_out_stream.getvalue(),
            stderr=capture_err_stream.getvalue(),
            error=str(e),
        )
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        capture_out_stream.close()
        capture_err_stream.close()
        return result


def reset_sandbox():
    global namespace
    namespace = {}
