"""Simple visibility metrics from a boolean 'visible' time series."""
import numpy as np


def fraction_true(mask):
    """Fraction of time steps where mask is True."""
    return float(np.mean(np.asarray(mask, dtype=bool)))


def runs(mask, value=True):
    """List of (start_index, length) for consecutive samples equal to value."""
    m = np.asarray(mask, dtype=bool) == value
    padded = np.concatenate(([False], m, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    starts, ends = edges[0::2], edges[1::2]
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def longest_run_hours(mask, step_hours, value=True):
    """Longest stretch (hours) where mask == value; 0 if it never happens."""
    r = runs(mask, value)
    return max((length for _, length in r), default=0) * step_hours