"""Exercise 9b"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import math

def exercise_9b(world, timestep, reset):
    """Exercise 9b"""
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive=4,
            amplitude=amplitude,
            phase_lag=phase_lag,
            turn=0,
            flag="9b"
            # ...
        )
        for phase_lag in np.linspace(0,4*math.pi,5)
        for amplitude in np.linspace(0.3,0.5,5)
        # for amplitudes in ...
        # for ...
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/example/simulation_{}.npz".format(simulation_i)
        )

