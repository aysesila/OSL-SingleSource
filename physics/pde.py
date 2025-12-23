import numpy as np

try:
    from numba import jit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

def _run_pde_steps_python(C, nx, ny, nt, dx, dt, vx, vy, k, source_mask, D):
    for _ in range(nt):
        Cold = C.copy()
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                d2x = (Cold[i, j+1] + Cold[i, j-1] - 2*Cold[i, j]) / dx**2
                d2y = (Cold[i+1, j] + Cold[i-1, j] - 2*Cold[i, j]) / dx**2

                # upwind-ish gradients (same as your notebook)
                grad_x = (Cold[i, j] - Cold[i, j-1]) / dx
                grad_y = (Cold[i, j] - Cold[i-1, j]) / dx

                C[i, j] = Cold[i, j] + dt * (
                    D*(d2x + d2y) - vx*grad_x - vy*grad_y - k*Cold[i, j] + source_mask[i, j]
                )
    return C

if _HAS_NUMBA:
    @jit(nopython=True)
    def _run_pde_steps_numba(C, nx, ny, nt, dx, dt, vx, vy, k, source_mask, D):
        for _ in range(nt):
            Cold = C.copy()
            for i in range(1, ny-1):
                for j in range(1, nx-1):
                    d2x = (Cold[i, j+1] + Cold[i, j-1] - 2*Cold[i, j]) / dx**2
                    d2y = (Cold[i+1, j] + Cold[i-1, j] - 2*Cold[i, j]) / dx**2
                    grad_x = (Cold[i, j] - Cold[i, j-1]) / dx
                    grad_y = (Cold[i, j] - Cold[i-1, j]) / dx
                    C[i, j] = Cold[i, j] + dt * (D*(d2x + d2y) - vx*grad_x - vy*grad_y - k*Cold[i, j] + source_mask[i, j])
        return C

class PDESolver:
    """
    2D advection-diffusion-decay with a Gaussian source term.
    """
    def __init__(self, wind_speed, wind_angle_deg, nx=30, ny=30, T=0.5, D=0.1, k=0.1, source_sigma_factor=2.0):
        self.wind_speed = float(wind_speed)
        self.wind_angle_deg = float(wind_angle_deg)
        self.nx = int(nx)
        self.ny = int(ny)

        self.D = float(D)
        self.k = float(k)

        self.dx = 1.0 / (self.nx - 1)

        ang = self.wind_angle_deg * np.pi / 180.0
        self.vx = self.wind_speed * np.cos(ang)
        self.vy = self.wind_speed * np.sin(ang)

        v_max = float(np.sqrt(self.vx**2 + self.vy**2))

        # CFL-ish stability
        dt_diff = 0.2 * self.dx**2 / (self.D + 1e-8)
        dt_adv = 0.8 * self.dx / (v_max + 1e-8) if v_max > 0 else dt_diff
        self.dt = min(dt_diff, dt_adv)

        self.nt = int(T / self.dt)
        self.source_sigma = source_sigma_factor * self.dx

    def _make_source(self, xs, ys):
        x = np.linspace(0, 1, self.nx)
        y = np.linspace(0, 1, self.ny)
        X, Y = np.meshgrid(x, y)
        sig = self.source_sigma
        src = np.exp(-((X - xs)**2 + (Y - ys)**2) / (2*sig**2))
        src = src / (np.sum(src) * self.dx**2 + 1e-10)
        return src

    def solve(self, xs, ys):
        C = np.zeros((self.ny, self.nx), dtype=np.float64)
        src = self._make_source(xs, ys)
        if _HAS_NUMBA:
            C = _run_pde_steps_numba(C, self.nx, self.ny, self.nt, self.dx, self.dt, self.vx, self.vy, self.k, src, self.D)
        else:
            C = _run_pde_steps_python(C, self.nx, self.ny, self.nt, self.dx, self.dt, self.vx, self.vy, self.k, src, self.D)
        return np.maximum(C, 0.0)
