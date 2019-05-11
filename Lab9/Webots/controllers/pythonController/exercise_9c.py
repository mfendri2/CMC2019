"""Exercise 9c"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9c(world, timestep, reset):
    """Exercise 9c"""
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive=4,
            turn=0,
            flag="9c",
            amplitude_gradient=[a_head,a_tail]
            # ...
        )
        for a_head in np.linspace(0.3,0.5,5)
        for a_tail in np.linspace(0.3,0.5,5)
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
