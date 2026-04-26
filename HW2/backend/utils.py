import pandas as pd
import numpy as np


def make_json_safe(obj):
    """
    Recursively convert pandas / numpy / Python objects
    into JSON-safe native Python types.
    """
    # None
    if obj is None:
        return None

    # NaN / pandas NA
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    # DataFrame
    if isinstance(obj, pd.DataFrame):
        return [make_json_safe(row) for row in obj.to_dict(orient="records")]

    # Series
    if isinstance(obj, pd.Series):
        return {str(k): make_json_safe(v) for k, v in obj.to_dict().items()}

    # pandas Index
    if isinstance(obj, pd.Index):
        return [make_json_safe(v) for v in obj.tolist()]

    # dict
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    # list / tuple / set
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    # numpy scalars
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    # ndarray
    if isinstance(obj, np.ndarray):
        return [make_json_safe(v) for v in obj.tolist()]

    # pandas timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    # native primitive
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # fallback
    return str(obj)
