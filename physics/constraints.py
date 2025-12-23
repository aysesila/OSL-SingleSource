import numpy as np

def in_bounds(pos, lo=0.1, hi=0.9) -> bool:
    xs, ys = float(pos[0]), float(pos[1])
    return (lo < xs < hi) and (lo < ys < hi)

def clip_to_bounds(pos, lo=0.1, hi=0.9):
    return np.clip(np.asarray(pos, dtype=np.float64), lo, hi)
