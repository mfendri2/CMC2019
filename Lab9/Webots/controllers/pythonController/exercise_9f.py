"""Exercise 9f"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import math


def exercise_9f(world, timestep, reset):
    """Exercise 9f"""
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive=2,
            turn=0,
            leg_body=leg_body,
            flag="9f",
            amplitude=0.3
            # ...
        )
        for leg_body in np.linspace(0,math.pi*2,50)
        #for amplitude in np.linspace(0,0.6,50)
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
            logs="./logs/9f/simulation_{}.npz".format(simulation_i)
        )

