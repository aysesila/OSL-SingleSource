import numpy as np

def sanity_oracle(solver, sensors, models, n_tests=10):
    """
    Test 1: Oracle consistency - can models reconstruct source from noiseless measurements?
    Expected: Physics-based methods should have very low error (~1e-4)
    """
    print("\n[Sanity 1] Oracle consistency")
    errs = {name: [] for name in models}

    for _ in range(n_tests):
        xs = np.random.uniform(0.2, 0.8)
        ys = np.random.uniform(0.2, 0.8)
        C = solver.solve(xs, ys)
        meas = sensors.measure(C, noise=0.0)

        for name, model in models.items():
            if name.startswith("MLP"):
                pred = model.predict(meas.reshape(1, -1))[0]
            else:
                pred = model.predict(meas)
            err = np.linalg.norm(pred - np.array([xs, ys]))
            errs[name].append(err)

    for k, v in errs.items():
        print(f"{k:15s}: {np.mean(v):.4e} ± {np.std(v):.4e}")


def sanity_sensor_permutation(model, solver, sensors, n_trials=5):
    """
    Test 2: Sensor permutation invariance for physics-based methods.
    Expected: Δ prediction ≈ 0 (physics doesn't care about sensor ordering)
    
    Two approaches:
    A) Permute both measurements AND sensor locations (physics stays consistent)
    B) Just permute measurements (tests if model is order-agnostic - should FAIL for physics models)
    
    We test approach A (the correct one for physics-based methods)
    """
    print("\n[Sanity 2] Sensor permutation invariance")
    
    deltas = []
    for _ in range(n_trials):
        xs, ys = np.random.uniform(0.3, 0.7, 2)
        C = solver.solve(xs, ys)
        
        # Get measurements with original sensor ordering
        meas_original = sensors.measure(C, noise=0.0)
        pred_original = model.predict(meas_original)
        
        # Create a permuted version
        perm = np.random.permutation(len(meas_original))
        
        # CRITICAL: Save original locations, then permute them
        original_locs = sensors.locations.copy()
        sensors.locations = sensors.locations[perm]
        
        # Re-measure with permuted sensor locations
        # (This should give same measurements but in different order)
        meas_permuted = sensors.measure(C, noise=0.0)
        pred_permuted = model.predict(meas_permuted)
        
        # Restore original sensor locations
        sensors.locations = original_locs
        
        # Predictions should be identical
        delta = np.linalg.norm(pred_original - pred_permuted)
        deltas.append(delta)
    
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    print(f"Δ prediction: {mean_delta:.6f} ± {std_delta:.6f}")
    
    if mean_delta < 1e-6:
        print("  ✓ PASS: Physics-based method is permutation invariant")
    elif mean_delta < 0.01:
        print("  ~ ACCEPTABLE: Small numerical errors")
    else:
        print("  ✗ FAIL: Should be near 0 for physics-based methods!")


def sanity_pde_residual(pinn, solver, n_tests=5):
    """
    Test 3: PDE residual should be lower for true source vs random source.
    Expected: r_true << r_random (physics is satisfied by true source)
    """
    print("\n[Sanity 3] PDE residual check")

    r_true_list = []
    r_rand_list = []
    
    for _ in range(n_tests):
        xs, ys = np.random.uniform(0.3, 0.7, 2)
        C_true = solver.solve(xs, ys)
        
        rand_x, rand_y = np.random.uniform(0.2, 0.8, 2)
        C_rand = solver.solve(rand_x, rand_y)
        
        r_true = pinn.compute_pde_residual(C_true)
        r_rand = pinn.compute_pde_residual(C_rand)
        
        r_true_list.append(r_true)
        r_rand_list.append(r_rand)
    
    print(f"Residual (true source): {np.mean(r_true_list):.6f} ± {np.std(r_true_list):.6f}")
    print(f"Residual (random):      {np.mean(r_rand_list):.6f} ± {np.std(r_rand_list):.6f}")
    
    ratio = np.mean(r_true_list) / np.mean(r_rand_list)
    print(f"Ratio (true/random):    {ratio:.4f}")
    if ratio > 0.5:
        print("  ⚠ WARNING: True source should have much lower residual!")


def sanity_pf_particles(pf, solver, sensors, n_particles_list, n_trials=10):
    """
    Test 4: PF error should decrease with more particles.
    Expected: Monotonic decrease in error as particles increase
    """
    print("\n[Sanity 4] PF particle count scaling")

    xs, ys = 0.6, 0.4
    C = solver.solve(xs, ys)
    meas = sensors.measure(C, noise=0.03)
    
    for n in n_particles_list:
        errs = []
        for trial in range(n_trials):
            np.random.seed(1000 + trial)  # Control randomness
            pf.n_particles = n
            pred = pf.predict(meas)
            err = np.linalg.norm(pred - np.array([xs, ys]))
            errs.append(err)
        
        mean_err = np.mean(errs)
        std_err = np.std(errs)
        print(f"Particles={n:4d} → error={mean_err:.4f} ± {std_err:.4f}")