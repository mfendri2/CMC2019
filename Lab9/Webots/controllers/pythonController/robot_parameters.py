"""Robot parameters"""

import numpy as np
import cmc_pylog as pylog
import math

class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def set_frequencies(self, parameters):
        self.freqs=np.ones(self.n_oscillators) #1 Hz


    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        smalldiag=np.diag(np.ones(int(self.n_oscillators_body/2))*10)
        largediag=np.diag(np.ones(self.n_oscillators-1)*10)
        largediag[self.n_oscillators_body][self.n_oscillators_body]=0
        
        self.coupling_weights[int(self.n_oscillators_body/2):self.n_oscillators_body,0:smalldiag.shape[1]]+=smalldiag; 
        self.coupling_weights[1:largediag.shape[0]+1,0:largediag.shape[1]]+=largediag;
        
        self.coupling_weights[0:smalldiag.shape[1],int(self.n_oscillators_body/2):self.n_oscillators_body]+=smalldiag; 
        self.coupling_weights[0:largediag.shape[1],1:largediag.shape[0]+1]+=largediag;


    def set_phase_bias(self, parameters):
        """Set phase bias"""
        smalldiag=np.diag(np.ones(int(self.n_oscillators_body/2))*math.pi)
        largediag=np.diag(np.ones(self.n_oscillators-1)*math.pi*2/8)
        largediag[self.n_oscillators_body][self.n_oscillators_body]=0
        
        self.phase_bias[int(self.n_oscillators_body/2):self.n_oscillators_body,0:smalldiag.shape[1]]+=smalldiag; 
        self.phase_bias[1:largediag.shape[0]+1,0:largediag.shape[1]]+=largediag;
        
        self.phase_bias[0:smalldiag.shape[1],int(self.n_oscillators_body/2):self.n_oscillators_body]+=smalldiag; 
        self.phase_bias[0:largediag.shape[1],1:largediag.shape[0]+1]+=largediag;


    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.rates+=20

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        self.nominal_amplitudes+=10

