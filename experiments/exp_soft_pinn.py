import numpy as np
from scipy.optimize import minimize

from physics.constraints import in_bounds

class SoftPINN:
    def __init__(self, solver, sensors, n_collocation=40, lambda_pde=5.0, bounds=(0.1, 0.9), nm_maxiter=30):
        self.solver = solver
        self.sensors = sensors
        self.n_collocation = int(n_collocation)
        self.lambda_pde = float(lambda_pde)
        self.bounds = bounds
        self.nm_maxiter = int(nm_maxiter)

    def compute_pde_residual(self, C):
        ny, nx = C.shape
        if nx < 3 or ny < 3:
            return 0.0

        cols = np.random.randint(1, nx - 1, self.n_collocation)
        rows = np.random.randint(1, ny - 1, self.n_collocation)

        dx = self.solver.dx
        residuals = []
        for i, j in zip(rows, cols):
            lap = (C[i+1, j] + C[i-1, j] + C[i, j+1] + C[i, j-1] - 4*C[i, j]) / (dx**2)
            gx = (C[i, j+1] - C[i, j-1]) / (2*dx)
            gy = (C[i+1, j] - C[i-1, j]) / (2*dx)
            r = self.solver.D * lap - (self.solver.vx * gx + self.solver.vy * gy) - self.solver.k * C[i, j]
            residuals.append(r * r)
        return float(np.mean(residuals)) if residuals else 0.0

    def loss(self, pos, measurements):
        lo, hi = self.bounds
        if not in_bounds(pos, lo, hi):
            return 1e8
        xs, ys = float(pos[0]), float(pos[1])
        C = self.solver.solve(xs, ys)
        pred = self.sensors.measure(C, noise=0.0)
        L_data = float(np.sum((pred - measurements) ** 2))
        L_pde = self.compute_pde_residual(C)
        return L_data + self.lambda_pde * L_pde

    def predict(self, measurements):
        best_x = np.array([0.5, 0.5], dtype=np.float64)
        best_f = np.inf
        starts = [np.array([0.5, 0.5]), np.array([0.3, 0.7]), np.array([0.7, 0.3])]
        for x0 in starts:
            res = minimize(lambda p: self.loss(p, measurements), x0, method="Nelder-Mead",
                           options={"maxiter": self.nm_maxiter})
            if res.fun < best_f:
                best_f = float(res.fun)
                best_x = np.array(res.x, dtype=np.float64)
        return best_x
