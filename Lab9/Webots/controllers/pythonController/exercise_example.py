"""Exercise example"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import math

def exercise_example(world, timestep, reset):
    """Exercise example"""
    # Parameters
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive=drive,
            #amplitudes=[1, 2, 3],
            phase_lag=2*math.pi,
            turn=0,
            # ...
        )
        for drive in [4,4,4,4,2,2,2,2]
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

