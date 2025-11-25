import numpy as np

def basic_statistics(values):
    """Return mean, std, min, max for a list/array of floats."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
