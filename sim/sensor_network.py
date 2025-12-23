import numpy as np

class SensorNetwork:
    def __init__(self, n_sensors=9):
        self.n_sensors = int(n_sensors)
        n_side = int(np.sqrt(self.n_sensors))
        assert n_side * n_side == self.n_sensors, "n_sensors must be a perfect square (e.g., 4, 9, 16)."

        locs = []
        for i in range(n_side):
            for j in range(n_side):
                locs.append([(i + 1) / (n_side + 1), (j + 1) / (n_side + 1)])
        self.locations = np.array(locs, dtype=np.float64)

    def measure(self, C, noise=0.05):
        ny, nx = C.shape
        measurements = []
        scale = float(np.max(C) + 1e-10)
        for sx, sy in self.locations:
            i = int(sx * (nx - 1))
            j = int(sy * (ny - 1))
            val = float(C[j, i])
            if noise and noise > 0:
                val += np.random.normal(0.0, noise * scale)
            measurements.append(max(0.0, val))
        return np.array(measurements, dtype=np.float64)
