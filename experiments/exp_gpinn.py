import numpy as np
from scipy.optimize import minimize, differential_evolution
from physics.constraints import in_bounds

class gPINN:
    """
    Improved gPINN with better optimization strategies
    - Multiple random starts
    - Optional global optimization
    - Adaptive lambda_pde based on loss magnitudes
    """
    def __init__(
        self, 
        solver, 
        sensors, 
        lambda_pde=10.0, 
        grad_weight=0.1, 
        bounds=(0.1, 0.9), 
        nm_maxiter=50,
        use_global_opt=False,
        n_starts=5
    ):
        self.solver = solver
        self.sensors = sensors
        self.lambda_pde = float(lambda_pde)
        self.grad_weight = float(grad_weight)
        self.bounds = bounds
        self.nm_maxiter = int(nm_maxiter)
        self.use_global_opt = use_global_opt
        self.n_starts = n_starts

    def compute_gpinn_term(self, C):
        """
        Compute gPINN physics loss: residual + gradient of residual
        """
        ny, nx = C.shape
        dx = self.solver.dx
        res = np.zeros_like(C, dtype=np.float64)

        # Compute PDE residual at interior points
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                # Laplacian
                lap = (C[i+1, j] + C[i-1, j] + C[i, j+1] + C[i, j-1] - 4*C[i, j]) / (dx**2)
                
                # Gradients (central difference)
                gx = (C[i, j+1] - C[i, j-1]) / (2*dx)
                gy = (C[i+1, j] - C[i-1, j]) / (2*dx)
                
                # PDE residual: D*∇²C - v·∇C - kC ≈ 0
                res[i, j] = self.solver.D * lap - (self.solver.vx * gx + self.solver.vy * gy) - self.solver.k * C[i, j]

        # Residual loss
        L_res = float(np.mean(res**2))
        
        # Gradient of residual (gPINN enhancement)
        gy, gx = np.gradient(res)
        L_grad = float(np.mean(gx**2 + gy**2))
        
        return L_res + self.grad_weight * L_grad

    def loss(self, pos, measurements, return_components=False):
        """
        Total loss: data + physics
        
        If return_components=True, returns (total, L_data, L_phys) for debugging
        """
        lo, hi = self.bounds
        if not in_bounds(pos, lo, hi):
            if return_components:
                return 1e8, 1e8, 0.0
            return 1e8
        
        xs, ys = float(pos[0]), float(pos[1])
        C = self.solver.solve(xs, ys)
        pred = self.sensors.measure(C, noise=0.0)
        
        L_data = float(np.sum((pred - measurements) ** 2))
        L_phys = self.compute_gpinn_term(C)
        
        total = L_data + self.lambda_pde * L_phys
        
        if return_components:
            return total, L_data, L_phys
        return total

    def predict(self, measurements, verbose=False):
        """
        Predict source location with improved optimization
        
        Parameters
        ----------
        measurements : np.ndarray
            Sensor measurements
        verbose : bool
            Print optimization details
        
        Returns
        -------
        best_x : np.ndarray
            Estimated source location
        """
        if self.use_global_opt:
            # Global optimization (slower but more robust)
            result = differential_evolution(
                lambda p: self.loss(p, measurements),
                bounds=[(self.bounds[0], self.bounds[1])] * 2,
                maxiter=100,
                popsize=15,
                seed=42
            )
            return np.array(result.x, dtype=np.float64)
        
        # Multi-start local optimization
        best_x = np.array([0.5, 0.5], dtype=np.float64)
        best_f = np.inf
        
        # Generate diverse starting points
        starts = [
            np.array([0.5, 0.5]),  # center
            np.array([0.3, 0.3]),  # bottom-left quadrant
            np.array([0.7, 0.7]),  # top-right quadrant
            np.array([0.3, 0.7]),  # top-left quadrant
            np.array([0.7, 0.3]),  # bottom-right quadrant
        ][:self.n_starts]
        
        for idx, x0 in enumerate(starts):
            res = minimize(
                lambda p: self.loss(p, measurements), 
                x0, 
                method="Nelder-Mead",
                options={"maxiter": self.nm_maxiter}
            )
            
            if verbose:
                total, L_data, L_phys = self.loss(res.x, measurements, return_components=True)
                print(f"  Start {idx+1}: f={res.fun:.6f} (data={L_data:.6f}, phys={L_phys:.6f})")
            
            if res.fun < best_f:
                best_f = float(res.fun)
                best_x = np.array(res.x, dtype=np.float64)
        
        if verbose:
            print(f"  Best solution: {best_x}, loss={best_f:.6f}")
        
        return best_x