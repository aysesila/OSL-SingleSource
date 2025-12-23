import numpy as np
from physics.pde import PDESolver
from training.trainer import make_dataset


def run_generalization(
    methods,
    sensors,
    base_solver,
    test_winds,
    n_test=20,
    noise=0.05,
    error_scale=10.0
):
    """
    Phase-2 generalization experiment.

    Parameters
    ----------
    methods : dict
        Dict[str, model], where model has .predict(measurements)
        (MLP, FourierMLP, PINN variants, PF, etc.)
    sensors : SensorNetwork
        Sensor network instance (shared across experiments)
    base_solver : PDESolver
        Solver used during training (for grid, D, k, T, etc.)
    test_winds : list of (wind_speed, wind_angle_deg)
        Out-of-distribution wind conditions
    n_test : int
        Number of test samples per wind condition
    noise : float
        Measurement noise level
    error_scale : float
        Spatial scaling (e.g. *10 for μm)

    Returns
    -------
    results : dict
        results[(wind_speed, wind_angle)][method_name] = np.ndarray of errors
    """

    results = {}

    for (wind_speed, wind_angle) in test_winds:
        # New solver with OOD wind, same physics otherwise
        solver = PDESolver(
            wind_speed=wind_speed,
            wind_angle_deg=wind_angle,
            nx=base_solver.nx,
            ny=base_solver.ny,
            T=base_solver.nt * base_solver.dt,
            D=base_solver.D,
            k=base_solver.k,
            source_sigma_factor=base_solver.source_sigma / base_solver.dx
        )

        # Generate test dataset under new wind
        X_test, Y_test = make_dataset(
            solver,
            sensors,
            n=n_test,
            noise=noise
        )

        wind_key = (float(wind_speed), float(wind_angle))
        results[wind_key] = {}

        # Evaluate each method
        for name, model in methods.items():
            errs = []

            for meas, ytrue in zip(X_test, Y_test):
                # Unified prediction interface
                if hasattr(model, "predict"):
                    if meas.ndim == 1:
                        pred = model.predict(meas)
                    else:
                        pred = model.predict(meas.reshape(1, -1))[0]
                else:
                    raise RuntimeError(f"Model {name} has no predict() method")

                err = np.linalg.norm(pred - ytrue) * error_scale
                errs.append(err)

            results[wind_key][name] = np.asarray(errs, dtype=np.float64)

    return results
