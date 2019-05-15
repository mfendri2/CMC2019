"""Exercise 9g"""
import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import math

def exercise_9g(world, timestep, reset):
    """Exercise 9f"""
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=20,
            drive=2,
            toggle="walk",
            flag="9g"
            # ...
        )
        #for leg_body in np.linspace(0,math.pi*2,50)
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/9g/simulation_{}.npz".format(simulation_i)
        )

