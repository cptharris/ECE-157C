"""
sandbox.py
"""

from typing import TypedDict, Optional


class ExecutionResult(TypedDict):
    stdout: str
    stderr: str
    error: Optional[str]


import sys
from io import StringIO
import traceback


def execute(code: str, namespace: dict) -> ExecutionResult:
    out_buf = StringIO()
    err_buf = StringIO()

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out_buf, err_buf

    try:
        exec(code, namespace)  # updates the passed dict in-place
        return ExecutionResult(
            stdout=out_buf.getvalue(), stderr=err_buf.getvalue(), error=None
        )
    except Exception as e:
        # write traceback into original stderr
        traceback.print_exc(file=orig_stderr)
        return ExecutionResult(
            stdout=out_buf.getvalue(), stderr=err_buf.getvalue(), error=str(e)
        )
    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        out_buf.close()
        err_buf.close()
