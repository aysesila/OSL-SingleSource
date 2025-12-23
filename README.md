# OSL-SingleSource
Single Source Single Molecule Odor Source Localization for Molecular Communication
Comparison of Physics Based Optimization, PINNs, Particle Filter and MLP methods

src/

  models/
  
    mlp.py              # MLP, training/inference forward pass
    
    fourier.py          # Fourier features wrapper OR FourierMLP subclass
    
  physics/
  
    pde.py              # forward model: concentration field, residuals, derivatives
    
    constraints.py      # hard constraints parameterization / boundary handling
    
  sim/
  
    sensor_network.py   # SensorNetwork (sensor positions, measurements)
    
    simulator.py        # dataset generation (sources, wind, noise, train/test splits)
    
  training/
  
    losses.py           # compute_losses() returns dict of loss terms
    
    sampling.py         # collocation sampling + RAR logic
    
    trainer.py          # train() loop (logs, checkpoints, early stop)
    
  baselines/
  
    particle_filter.py  # ParticleFilter baseline
    
  plots/
  
    figures.py          
    
  experiments/
  
    registry.py         # central experiment definitions (configs)
    
    run_experiment.py   # run one experiment by name
    
  utils/
  
    config.py           # dataclasses / yaml loading
    
    seed.py             # seeding + device helpers
    
    metrics.py          # error metrics + CDF computation
    
scripts/run_all.py            # runs the full comparison matrix and saves outputs
  
notebooks/demo.ipynb            # thin: imports + calls runner

