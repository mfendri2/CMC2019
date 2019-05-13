"""Simulation parameters"""
import math

class SimulationParameters(dict):
    """Simulation parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 10
        self.n_legs_joints = 4
        self.simulation_duration = 30
        self.phase_lag = 2*math.pi
        self.amplitude_gradient = [0.3,0.3]
        self.drive=4
        self.flag="example"
        self.back=False
        self.turn=0
        self.amplitude=0
        self.leg_body=math.pi
        # Feel free to add more parameters (ex: MLR drive)
        #self.drive_mlr = ...
        # ...
        # Update object with provided keyword arguments
        self.update(kwargs)  # NOTE: This overrides the previous declarations

