import numpy as np
from .exp_soft_pinn import SoftPINN

class RAR_PINN(SoftPINN):
    """
    Residual-based adaptive refinement: update collocation points by picking highest residual locations.
    """
    def __init__(self, *args, update_freq=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_freq = int(update_freq)
        self.eval_count = 0

        # initialize collocation points on grid
        n = int(np.sqrt(self.n_collocation))
        x = np.linspace(0.15, 0.85, n)
        y = np.linspace(0.15, 0.85, n)
        X, Y = np.meshgrid(x, y)
        self.collocation_points = np.column_stack([X.ravel(), Y.ravel()])

    def compute_pde_residual(self, C):
        ny, nx = C.shape
        dx = self.solver.dx

        residuals = []
        for x_c, y_c in self.collocation_points:
            j = int(x_c * (nx - 1))
            i = int(y_c * (ny - 1))
            if not (0 < i < ny-1 and 0 < j < nx-1):
                continue
            lap = (C[i+1, j] + C[i-1, j] + C[i, j+1] + C[i, j-1] - 4*C[i, j]) / (dx**2)
            gx = (C[i, j+1] - C[i, j-1]) / (2*dx)
            gy = (C[i+1, j] - C[i-1, j]) / (2*dx)
            r = self.solver.D * lap - (self.solver.vx * gx + self.solver.vy * gy) - self.solver.k * C[i, j]
            residuals.append(r * r)
        return float(np.mean(residuals)) if residuals else 0.0

    def _update_points(self, C):
        ny, nx = C.shape
        dx = self.solver.dx
        res_map = np.zeros((ny, nx), dtype=np.float64)

        for i in range(1, ny-1):
            for j in range(1, nx-1):
                lap = (C[i+1, j] + C[i-1, j] + C[i, j+1] + C[i, j-1] - 4*C[i, j]) / (dx**2)
                gx = (C[i, j+1] - C[i, j-1]) / (2*dx)
                gy = (C[i+1, j] - C[i-1, j]) / (2*dx)
                r = abs(self.solver.D * lap - (self.solver.vx * gx + self.solver.vy * gy) - self.solver.k * C[i, j])
                res_map[i, j] = r

        flat = np.argsort(res_map.ravel())[-len(self.collocation_points):]
        new_pts = []
        for idx in flat:
            i, j = np.unravel_index(idx, (ny, nx))
            new_pts.append([j/(nx-1), i/(ny-1)])
        self.collocation_points = np.array(new_pts, dtype=np.float64)

    def loss(self, pos, measurements):
        # call base loss but update points periodically
        xs, ys = float(pos[0]), float(pos[1])
        C = self.solver.solve(xs, ys)
        pred = self.sensors.measure(C, noise=0.0)
        L_data = float(np.sum((pred - measurements) ** 2))
        L_pde = self.compute_pde_residual(C)

        self.eval_count += 1
        if self.eval_count % self.update_freq == 0:
            self._update_points(C)

        return L_data + self.lambda_pde * L_pde
