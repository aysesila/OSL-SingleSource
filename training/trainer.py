import numpy as np
from .sampling import sample_sources

def make_dataset(solver, sensors, n, lo=0.2, hi=0.8, noise=0.05):
    src = sample_sources(n, lo=lo, hi=hi)
    X = []
    Y = []
    for xs, ys in src:
        C = solver.solve(xs, ys)
        meas = sensors.measure(C, noise=noise)
        X.append(meas)
        Y.append([xs, ys])
    return np.array(X), np.array(Y)

def train_val_split(X, Y, frac=0.8):
    n = len(X)
    split = int(frac * n)
    return X[:split], Y[:split], X[split:], Y[split:]
