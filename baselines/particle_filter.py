import numpy as np
from physics.constraints import clip_to_bounds

class ParticleFilter:
    """
    Improved Particle Filter with better numerical stability
    """
    def __init__(
        self,
        solver,
        sensors,
        n_particles=200,
        bounds=(0.1, 0.9),
        sigma_likelihood=0.05,
        n_iters=3,
        jitter=0.02,
        eps=1e-12,
    ):
        self.solver = solver
        self.sensors = sensors
        self.n_particles = int(n_particles)
        self.bounds = bounds
        self.sigma_likelihood = float(sigma_likelihood)
        self.n_iters = int(n_iters)
        self.jitter = float(jitter)
        self.eps = eps

    def _init_particles(self):
        """Initialize particles uniformly in bounds"""
        lo, hi = self.bounds
        return np.random.uniform(lo, hi, size=(self.n_particles, 2))

    def _log_likelihood(self, particles, measurements):
        """Compute log-likelihood for each particle"""
        logw = np.zeros(len(particles))
        for i, (xs, ys) in enumerate(particles):
            C = self.solver.solve(xs, ys)
            pred = self.sensors.measure(C, noise=0.0)
            err = np.sum((pred - measurements) ** 2)
            # Negative log-likelihood (Gaussian assumption)
            logw[i] = -err / (2 * self.sigma_likelihood**2)
        return logw

    def _normalize_weights(self, logw):
        """
        Normalize log-weights using log-sum-exp trick for numerical stability
        """
        # Subtract max for numerical stability
        logw_shifted = logw - np.max(logw)
        w = np.exp(logw_shifted)
        
        # Sum and normalize
        s = np.sum(w)
        
        # Fallback to uniform if numerical issues
        if not np.isfinite(s) or s < self.eps:
            return np.ones_like(w) / len(w)
        
        return w / s

    def _effective_sample_size(self, weights):
        """Compute effective sample size (ESS)"""
        return 1.0 / np.sum(weights**2)

    def _resample(self, particles, weights):
        """Systematic resampling (more stable than multinomial)"""
        n = len(particles)
        
        # Cumulative sum of weights
        cumsum = np.cumsum(weights)
        
        # Systematic resampling positions
        positions = (np.arange(n) + np.random.uniform()) / n
        
        # Find indices
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, n-1)
        
        return particles[indices]

    def predict(self, measurements):
        """
        Run particle filter to estimate source location
        
        Returns
        -------
        estimate : np.ndarray
            Weighted mean of particles (better than simple mean)
        """
        particles = self._init_particles()
        
        for iter_idx in range(self.n_iters):
            # Compute weights
            logw = self._log_likelihood(particles, measurements)
            weights = self._normalize_weights(logw)
            
            # Check effective sample size
            ess = self._effective_sample_size(weights)
            
            # Only resample if ESS is low (adaptive resampling)
            if ess < self.n_particles / 2:
                particles = self._resample(particles, weights)
                # After resampling, weights are uniform
                weights = np.ones(self.n_particles) / self.n_particles
            
            # Jitter / rejuvenation (with adaptive scaling)
            # Reduce jitter in later iterations
            jitter_scale = self.jitter * (0.5 ** iter_idx)
            particles += jitter_scale * np.random.randn(*particles.shape)
            
            # Enforce bounds
            particles = clip_to_bounds(particles, *self.bounds)
        
        # Final weight computation
        logw = self._log_likelihood(particles, measurements)
        weights = self._normalize_weights(logw)
        
        # Return weighted mean (better than simple mean)
        estimate = np.average(particles, axis=0, weights=weights)
        
        return estimate