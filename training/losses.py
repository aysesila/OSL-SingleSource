import numpy as np

def mse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.mean((a - b) ** 2))

def l2(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sum((a - b) ** 2))
