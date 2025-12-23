import matplotlib.pyplot as plt
import numpy as np

def plot_fig9_optimization_trials(results, outpath="Fig9_Optimization_Trials.png"):
    methods = list(results.keys())
    means = [results[m]['mean'] for m in methods]
    stds = [results[m]['std'] for m in methods]
    colors = ['#BDC3C7', '#3498DB', '#9B59B6', '#E67E22', '#2ECC71', '#E74C3C', '#2C3E50']

    plt.figure(figsize=(12, 6))
    bars = plt.bar(methods, means, yerr=stds, capsize=8, color=colors[:len(methods)], alpha=0.8, edgecolor='black')
    plt.ylabel('Localization Error (μm)', fontsize=14, fontweight='bold')
    plt.title('Impact of Advanced PINN Methods', fontsize=16, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, mean + std + 0.05, f'{mean:.2f}',
                 ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"✓ Saved: {outpath}")

def plot_training_curves(train_losses, val_losses, outpath="Fig1_Training_Curves.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', alpha=0.8, linewidth=2)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8, linewidth=2)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    plt.title('Training Curves', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"✓ Saved: {outpath}")


def plot_generalization_bars(
    gen_results,
    outpath="Fig5a_Generalization_Bars.png"
):
    """
    Bar plot of mean localization error under OOD wind conditions.
    gen_results[(wind_speed, wind_angle)][method] = errors
    """

    wind_keys = list(gen_results.keys())
    methods = list(next(iter(gen_results.values())).keys())

    means = {m: [] for m in methods}
    stds = {m: [] for m in methods}

    for w in wind_keys:
        for m in methods:
            errs = gen_results[w][m]
            means[m].append(np.mean(errs))
            stds[m].append(np.std(errs))

    x = np.arange(len(wind_keys))
    width = 0.12

    plt.figure(figsize=(14, 6))
    for i, m in enumerate(methods):
        plt.bar(
            x + i * width,
            means[m],
            width,
            yerr=stds[m],
            capsize=4,
            label=m,
            alpha=0.85
        )

    labels = [f"v={w[0]:.3f}, θ={w[1]:.0f}°" for w in wind_keys]
    plt.xticks(x + width * (len(methods) - 1) / 2, labels, fontsize=11)
    plt.ylabel("Localization Error (μm)", fontsize=13, fontweight="bold")
    plt.title("Phase-2 Generalization Under OOD Wind", fontsize=15, fontweight="bold")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"✓ Saved: {outpath}")

def plot_error_cdf(
    gen_results,
    outpath="Fig7b_Error_CDF.png"
):
    """
    CDF of localization error across all OOD conditions.
    """

    plt.figure(figsize=(8, 6))

    # Aggregate across all winds
    methods = list(next(iter(gen_results.values())).keys())
    all_errs = {m: [] for m in methods}

    for w in gen_results:
        for m in methods:
            all_errs[m].extend(gen_results[w][m])

    for m, errs in all_errs.items():
        errs = np.sort(np.asarray(errs))
        cdf = np.linspace(0, 1, len(errs))
        plt.plot(errs, cdf, linewidth=2, label=m)

    plt.xlabel("Localization Error (μm)", fontsize=13, fontweight="bold")
    plt.ylabel("CDF", fontsize=13, fontweight="bold")
    plt.title("Error Distribution Under OOD Conditions", fontsize=15, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"✓ Saved: {outpath}")



def plot_spatial_error_maps(
    methods,
    solver,
    sensors,
    n_grid=15,
    noise=0.0,
    error_scale=10.0,
    outpath="Fig3_Spatial_Error_Maps.png"
):
    """
    Spatial error heatmaps over the source domain.
    """

    xs = np.linspace(0.2, 0.8, n_grid)
    ys = np.linspace(0.2, 0.8, n_grid)
    X, Y = np.meshgrid(xs, ys)

    n_methods = len(methods)
    #fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), sharey=True)

    fig, axes = plt.subplots(
    1,
    n_methods,
    figsize=(4 * n_methods, 4),
    sharey=True,
    constrained_layout=True
)

    if n_methods == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, methods.items()):
        err_map = np.zeros_like(X)

        for i in range(n_grid):
            for j in range(n_grid):
                xs0, ys0 = X[i, j], Y[i, j]
                C = solver.solve(xs0, ys0)
                meas = sensors.measure(C, noise=noise)

                if meas.ndim == 1:
                    pred = model.predict(meas)
                else:
                    pred = model.predict(meas.reshape(1, -1))[0]

                err_map[i, j] = np.linalg.norm(pred - np.array([xs0, ys0])) * error_scale

        im = ax.imshow(
            err_map,
            origin="lower",
            extent=[0.2, 0.8, 0.2, 0.8],
            cmap="viridis"
        )
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_aspect("equal")

    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04, label="Error (μm)")
    plt.suptitle("Spatial Localization Error Maps", fontsize=15, fontweight="bold")
    #plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"✓ Saved: {outpath}")


def plot_concentration_fields(
    solver,
    sources,
    outpath="Fig6_Concentration_Fields.png"
):
    """
    Plot PDE concentration fields for selected source locations.
    """

    n = len(sources)
    #fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    fig, axes = plt.subplots(
    1,
    n,
    figsize=(4 * n, 4),
    sharey=True,
    constrained_layout=True
)
    if n == 1:
        axes = [axes]

    for ax, (xs, ys) in zip(axes, sources):
        C = solver.solve(xs, ys)
        im = ax.imshow(C, origin="lower", cmap="inferno")
        ax.set_title(f"Source ({xs:.2f}, {ys:.2f})", fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04, label="Concentration")
    plt.suptitle("Advection–Diffusion Concentration Fields", fontsize=15, fontweight="bold")
    #plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"✓ Saved: {outpath}")


def plot_uncertainty_analysis(
    model,
    solver,
    sensors,
    true_source,
    noise_levels,
    n_trials=50,
    error_scale=10.0,
    outpath="Fig8_Uncertainty_Analysis.png"
):
    """
    Prediction uncertainty vs noise level.
    """

    xs_true, ys_true = true_source
    C = solver.solve(xs_true, ys_true)

    means = []
    stds = []

    for noise in noise_levels:
        preds = []
        for _ in range(n_trials):
            meas = sensors.measure(C, noise=noise)
            pred = model.predict(meas)
            preds.append(pred)

        preds = np.array(preds)
        errs = np.linalg.norm(preds - np.array(true_source), axis=1) * error_scale
        means.append(np.mean(errs))
        stds.append(np.std(errs))

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        noise_levels,
        means,
        yerr=stds,
        fmt="o-",
        capsize=4,
        linewidth=2
    )
    plt.xlabel("Measurement Noise Level", fontsize=12, fontweight="bold")
    plt.ylabel("Localization Error (μm)", fontsize=12, fontweight="bold")
    plt.title("Uncertainty Analysis", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"✓ Saved: {outpath}")
