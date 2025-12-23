import numpy as np

def sample_sources(n, lo=0.2, hi=0.8):
    xs = np.random.uniform(lo, hi, size=n)
    ys = np.random.uniform(lo, hi, size=n)
    return np.column_stack([xs, ys])
