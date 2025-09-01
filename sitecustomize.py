# sitecustomize.py
import numpy as _np

# pandas_ta (<=0.3.14) expects aliases removed in NumPy 2.x
if not hasattr(_np, "NaN"):
    _np.NaN = _np.nan
if not hasattr(_np, "Inf"):
    _np.Inf = _np.inf
