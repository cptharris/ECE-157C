import pandas as pd
import numpy as np
import json
import base64


def make_json_safe(obj):
    if obj is None:
        return None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, pd.DataFrame):
        return [make_json_safe(row) for row in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return {str(k): make_json_safe(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, pd.Index, np.ndarray)):
        return [make_json_safe(v) for v in list(obj)]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def describe_dataframe_like(data) -> str:
    df = (
        data
        if isinstance(data, pd.DataFrame)
        else pd.DataFrame(data if isinstance(data, list) else [data])
    )
    return json.dumps(
        {
            "row_count": len(df),
            "columns": df.dtypes.astype(str).to_dict(),
            "sample": df.head(2).to_dict(orient="records"),
        },
        indent=None,
    )[:3000]


def sanitize_plotly(obj):
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            return np.frombuffer(
                base64.b64decode(obj["bdata"]), dtype=np.dtype(obj["dtype"])
            ).tolist()
        return {k: sanitize_plotly(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_plotly(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return obj
