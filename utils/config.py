from dataclasses import dataclass

@dataclass
class PDEConfig:
    nx: int = 30
    ny: int = 30
    T: float = 0.5
    D: float = 0.1
    k: float = 0.1
    source_sigma_factor: float = 2.0  # sigma = factor * dx

@dataclass
class ExperimentConfig:
    seed: int = 42
    train_wind: float = 0.001
    train_angle_deg: float = 45.0
    n_sensors: int = 9

    n_train: int = 2000
    n_test: int = 20
    train_noise: float = 0.05
    test_noise: float = 0.05

    mlp_epochs: int = 150
    mlp_lr: float = 0.01
   

    # inverse optimization
    nm_maxiter_soft: int = 30
    nm_maxiter_hard: int = 40
    bounds_lo: float = 0.1
    bounds_hi: float = 0.9
