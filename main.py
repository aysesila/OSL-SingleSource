import numpy as np
from scipy.optimize import minimize

from utils.config import PDEConfig, ExperimentConfig
from physics.pde import PDESolver
from sim.sensor_network import SensorNetwork
from physics.constraints import in_bounds

from models.mlp import MLP
from models.fourier import FourierMLP

from training.trainer import make_dataset, train_val_split
from plots.figures import plot_fig9_optimization_trials, plot_training_curves, plot_generalization_bars, plot_error_cdf, plot_spatial_error_maps, plot_concentration_fields, plot_uncertainty_analysis

from experiments.exp_soft_pinn import SoftPINN
from experiments.exp_rar import RAR_PINN
from experiments.exp_gpinn import gPINN
from experiments.exp_generalization import run_generalization
from baselines.particle_filter import ParticleFilter

from experiments.exp_sanity import sanity_oracle, sanity_sensor_permutation, sanity_pde_residual, sanity_pf_particles



class HardPINN:
    """
    Direct inverse solve: minimize sensor MSE wrt source position, no PDE residual term.
    """
    def __init__(self, solver, sensors, bounds=(0.1, 0.9), nm_maxiter=40):
        self.solver = solver
        self.sensors = sensors
        self.bounds = bounds
        self.nm_maxiter = nm_maxiter

    def loss(self, pos, measurements):
        lo, hi = self.bounds
        if not in_bounds(pos, lo, hi):
            return 1e8
        xs, ys = float(pos[0]), float(pos[1])
        C = self.solver.solve(xs, ys)
        pred = self.sensors.measure(C, noise=0.0)
        return float(np.sum((pred - measurements) ** 2))

    def predict(self, measurements):
        best_x = np.array([0.5, 0.5], dtype=np.float64)
        best_f = np.inf
        starts = [np.array([0.5, 0.5]), np.array([0.2, 0.2]), np.array([0.8, 0.8])]
        for x0 in starts:
            res = minimize(lambda p: self.loss(p, measurements), x0, method="Nelder-Mead",
                           options={"maxiter": self.nm_maxiter})
            if res.fun < best_f:
                best_f = float(res.fun)
                best_x = np.array(res.x, dtype=np.float64)
        return best_x


def eval_method(method_name, method_obj, X_test, Y_test):
    errs = []
    for meas, ytrue in zip(X_test, Y_test):
        if method_name.startswith("MLP"):
            pred = method_obj.predict(meas.reshape(1, -1))[0]
        else:
            pred = method_obj.predict(meas)
        err_um = np.linalg.norm(pred - ytrue) * 10.0  # your convention
        errs.append(err_um)
    errs = np.array(errs)
    return {"mean": float(np.mean(errs)), "std": float(np.std(errs)), "all": errs}


def main():
    pde_cfg = PDEConfig()
    exp_cfg = ExperimentConfig()

    np.random.seed(exp_cfg.seed)

    solver = PDESolver(
        wind_speed=exp_cfg.train_wind,
        wind_angle_deg=exp_cfg.train_angle_deg,
        nx=pde_cfg.nx,
        ny=pde_cfg.ny,
        T=pde_cfg.T,
        D=pde_cfg.D,
        k=pde_cfg.k,
        source_sigma_factor=pde_cfg.source_sigma_factor
    )
    sensors = SensorNetwork(n_sensors=exp_cfg.n_sensors)

    print("\n[1] Generate dataset")
    X, Y = make_dataset(solver, sensors, n=exp_cfg.n_train, noise=exp_cfg.train_noise)
    Xtr, Ytr, Xva, Yva = train_val_split(X, Y, frac=0.8)

    print("\n[2] Train MLP baselines")
    mlp = MLP(input_dim=exp_cfg.n_sensors, seed=0)
    mlp.train(Xtr, Ytr, Xva, Yva, epochs=exp_cfg.mlp_epochs, lr=exp_cfg.mlp_lr, verbose_every=25)

    fmlp = FourierMLP(input_dim=exp_cfg.n_sensors, mapping_size=32, scale=10.0, seed=0)
    fmlp.train(Xtr, Ytr, Xva, Yva, epochs=exp_cfg.mlp_epochs, lr=exp_cfg.mlp_lr, verbose_every=25)

    plot_training_curves(mlp.train_losses, mlp.val_losses, outpath="Fig1_Training_Curves.png")

    print("\n[3] Build test set")
    np.random.seed(999)
    Xtest, Ytest = make_dataset(solver, sensors, n=exp_cfg.n_test, noise=exp_cfg.test_noise)

    print("\n[4] Initialize inverse solvers")
    soft = SoftPINN(solver, sensors, lambda_pde=5.0, bounds=(exp_cfg.bounds_lo, exp_cfg.bounds_hi), nm_maxiter=exp_cfg.nm_maxiter_soft)
    rar = RAR_PINN(solver, sensors, lambda_pde=5.0, bounds=(exp_cfg.bounds_lo, exp_cfg.bounds_hi), nm_maxiter=exp_cfg.nm_maxiter_soft)
    #gp = gPINN(solver, sensors, lambda_pde=10.0, grad_weight=0.1, bounds=(exp_cfg.bounds_lo, exp_cfg.bounds_hi), nm_maxiter=50)
    gp = gPINN(
    solver, 
    sensors, 
    lambda_pde=10.0,      # Keep same
    grad_weight=0.1,      # Keep same
    bounds=(exp_cfg.bounds_lo, exp_cfg.bounds_hi),
    nm_maxiter=100,       # INCREASED from 50
    use_global_opt=False, # Set True for even better results (but slower)
    n_starts=5            # More starting points
)
   
    hard = HardPINN(solver, sensors, bounds=(exp_cfg.bounds_lo, exp_cfg.bounds_hi), nm_maxiter=exp_cfg.nm_maxiter_hard)

    methods = {
        "MLP (Base)": mlp,
        "MLP (Fourier)": fmlp,
        "Soft PINN": soft,
        "PINN + RAR": rar,
        "PINN + gPINN": gp,
        "Hard PINN": hard,
    }

    pf = ParticleFilter(
    solver,
    sensors,
    n_particles=500,
    sigma_likelihood=0.05,
    n_iters=3,
    jitter=0.02,
)


    methods["Particle Filter"] = pf

    print("\n[5] Evaluate")
    stats = {}
    for name, obj in methods.items():
        s = eval_method(name, obj, Xtest, Ytest)
        stats[name] = {"mean": s["mean"], "std": s["std"]}
        print(f"  {name:<14} : {s['mean']:.3f} ± {s['std']:.3f} μm")

    print("\n[6] Plot Fig9")
    plot_fig9_optimization_trials(stats, outpath="Fig9_Optimization_Trials.png")


    gen_results = run_generalization(
    methods,
    sensors,
    solver,
    test_winds=[(0.002, 30), (0.002, 90), (0.003, 45)],
    n_test=20,
    noise=exp_cfg.test_noise
)

    plot_generalization_bars(gen_results)
    plot_error_cdf(gen_results)

    plot_spatial_error_maps(
    {
        "MLP": mlp,
        "gPINN": gp,
        "Hard PINN": hard,
        "PF": pf,
    },
    solver,
    sensors
)

    plot_concentration_fields(
    solver,
    sources=[(0.3, 0.3), (0.5, 0.6), (0.7, 0.4)]
)

    plot_uncertainty_analysis(
    model=gp,               # or hard / PF
    solver=solver,
    sensors=sensors,
    true_source=(0.5, 0.5),
    noise_levels=[0.01, 0.03, 0.05, 0.1]
)

    print("\n======================")
    print("Running sanity checks")
    print("======================")

    # 1) Oracle consistency
    print("\n[Test 1] Oracle Consistency")
    print("Expected: Physics-based methods should reconstruct source perfectly (error ~1e-4)")
    sanity_oracle(
        solver,
        sensors,
        {
            "Hard PINN": hard,
            "gPINN": gp,
            "Particle Filter": pf,
            "MLP": mlp,
        },
    )

    # 2) Sensor permutation invariance
    print("\n[Test 2] Sensor Permutation Invariance")
    print("Expected: Physics-based methods should be invariant to sensor ordering (Δ ~0)")
    sanity_sensor_permutation(
        hard,
        solver,
        sensors,
        n_trials=5
    )

    

    # 4) Particle Filter scaling
    print("\n[Test 4] Particle Filter Particle Count Scaling")
    print("Expected: Error should decrease monotonically with more particles")
    sanity_pf_particles(
        pf,
        solver,
        sensors,
        n_particles_list=[50, 100, 200, 500, 1000],
        n_trials=10
    )


    # ========================================
    # ADDITIONAL: Test gPINN in verbose mode
    # ========================================
    print("\n======================")
    print("gPINN Diagnostic Test")
    print("======================")

    # Test one prediction with verbose output
    xs_test, ys_test = 0.5, 0.5
    C_test = solver.solve(xs_test, ys_test)
    meas_test = sensors.measure(C_test, noise=0.0)

    print(f"\nTrue source: ({xs_test:.3f}, {ys_test:.3f})")
    print("Running gPINN optimization with verbose output:")
    pred_gp = gp.predict(meas_test, verbose=True)
    err_gp = np.linalg.norm(pred_gp - np.array([xs_test, ys_test]))
    print(f"gPINN prediction: ({pred_gp[0]:.3f}, {pred_gp[1]:.3f})")
    print(f"Error: {err_gp:.6f}")
    print(f"Expected: < 0.001 for oracle test")

if __name__ == "__main__":
    main()
