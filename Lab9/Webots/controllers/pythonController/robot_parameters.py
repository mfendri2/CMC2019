"""Robot parameters"""

import numpy as np
import cmc_pylog as pylog
import math
from numpy import genfromtxt

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
        #self.turning=parameters.turn
        self.update(parameters)


    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i
        
        
#    def turn(self,value):
#        print(turning)
#        if value:
#            self.flag="9d1g"
#            self.d=self.d-self.turning
#            self.set_frequencies()
#            self.set_nominal_amplitudes()
#            self.flag="9d1"
#        else:
#            self.flag="9d1s"
#            self.d=self.d+self.turning
#            self.set_frequencies()
#            self.set_nominal_amplitudes()
#            self.flag="9d1"

            
            

    def set_frequencies(self,parameters):
        d=parameters.drive
        f_b=(0.2*d+0.3)
        f_l=(0.0+0.2*d)
        if d<1:
            f_l=0
            f_b=0
        if d>3:
            f_l=0
        if d>5:
            f_b=0
        self.freqs=np.ones(self.n_oscillators) 
        self.freqs[0:self.n_oscillators_body]=f_b
        self.freqs[self.n_oscillators_body:self.n_oscillators]=f_l
        
        if parameters.turn !=0 and parameters.turning:
            d1=parameters.drive+parameters.turn
            d2=parameters.drive-parameters.turn
            f_b_1=(0.196+0.065*d1)
            f_b_2=(0.196+0.065*d2)
            if d1<1:
                f_b_1=0
            if d1>5:
                f_b_1=0
            if d2<1:
                f_b_2=0
            if d2>5:
                f_b_2=0      
            print(f_b_1,f_b_2,parameters.turn)
            self.freqs=np.ones(self.n_oscillators) 
            self.freqs[0:self.n_oscillators_body//2]=f_b_1
            self.freqs[self.n_oscillators_body//2:self.n_oscillators_body]=f_b_2
            self.freqs[self.n_oscillators_body:self.n_oscillators]=f_l

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
#        smalldiag=np.diag(np.ones(int(self.n_oscillators_body/2))*10)
#        largediag=np.diag(np.ones(self.n_oscillators-1)*10)
#        largediag[self.n_oscillators_body/2][self.n_oscillators_body/2]=0
#        
#        self.coupling_weights[int(self.n_oscillators_body/2):self.n_oscillators_body,0:smalldiag.shape[1]]+=smalldiag; 
#        self.coupling_weights[1:largediag.shape[0]+1,0:largediag.shape[1]]+=largediag;
#        
#        self.coupling_weights[0:smalldiag.shape[1],int(self.n_oscillators_body/2):self.n_oscillators_body]+=smalldiag; 
#        self.coupling_weights[0:largediag.shape[1],1:largediag.shape[0]+1]+=largediag;
        self.coupling_weights=genfromtxt("array.csv",delimiter=",")


    def set_phase_bias(self, parameters):
        """Set phase bias"""
#        smalldiag=np.diag(np.ones(int(self.n_oscillators_body/2))*math.pi)

#        self.phase_bias[int(self.n_oscillators_body/2):self.n_oscillators_body,0:smalldiag.shape[1]]+=smalldiag; 
#        self.phase_bias[1:largediag.shape[0]+1,0:largediag.shape[1]]+=largediag;
#        
#        self.phase_bias[0:smalldiag.shape[1],int(self.n_oscillators_body/2):self.n_oscillators_body]+=smalldiag; 
#        self.phase_bias[0:largediag.shape[1],1:largediag.shape[0]+1]+=largediag;
        self.phase_bias=genfromtxt("array_phase.csv",delimiter=",")
        print(parameters.phase_lag)
        largediag=np.diag(np.ones(self.n_oscillators_body-1)*parameters.phase_lag/10)
        largediag[int(self.n_oscillators_body/2-1)][int(self.n_oscillators_body/2-1)]=0
        
        #Setting up the phases biases between limb and body oscillators. 
        num_assigned=self.n_oscillators_body//self.n_oscillators_legs
        phases=np.ones(num_assigned)*parameters.leg_body
        
        for i in range(self.n_oscillators_legs):
            if i%2==0:
                if i==0:
                    self.phase_bias[self.n_oscillators_body+i,(i)*num_assigned:(i)*(num_assigned)+num_assigned]+=phases
                    self.phase_bias[(i)*num_assigned:(i)*(num_assigned)+num_assigned,self.n_oscillators_body+i]+=phases.T
                else:
                    self.phase_bias[self.n_oscillators_body+i,(i-1)*num_assigned:(i-1)*(num_assigned)+num_assigned]+=phases
                    self.phase_bias[(i-1)*num_assigned:(i-1)*(num_assigned)+num_assigned,self.n_oscillators_body+i]+=phases.T
            else:
                if i==3:
                    self.phase_bias[self.n_oscillators_body+i,(i)*num_assigned:(i)*(num_assigned)+num_assigned]+=phases.T
                    self.phase_bias[(i)*num_assigned:(i)*(num_assigned)+num_assigned,self.n_oscillators_body+i]+=phases.T
                else:
                    self.phase_bias[self.n_oscillators_body+i,(i+1)*num_assigned:(i+1)*(num_assigned)+num_assigned]+=phases
                    self.phase_bias[(i+1)*num_assigned:(i+1)*(num_assigned)+num_assigned,self.n_oscillators_body+i]+=phases.T

        
        if parameters.back== False:
            self.phase_bias[1:largediag.shape[0]+1,0:largediag.shape[1]]+=largediag;
            self.phase_bias[0:largediag.shape[1],1:largediag.shape[0]+1]-=largediag;

        
        if parameters.back==True:
            self.phase_bias[1:largediag.shape[0]+1,0:largediag.shape[1]]-=largediag;
            self.phase_bias[0:largediag.shape[1],1:largediag.shape[0]+1]+=largediag;
        np.savetxt("test_phase.csv", self.phase_bias,delimiter=",")
    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.rates+=20
        print(np.shape(self.rates))
        

    def set_nominal_amplitudes(self,parameters):
        """Set nominal amplitudes"""
        d=parameters.drive
        R_b=(0.196+0.065*d)
        R_l=(0.131+0.131*d)
        
        if d<1:
            R_l=0
            R_b=0
        if d>3:
            R_l=0
        if d>5:
            R_b=0
        if parameters.flag=="9b" or parameters.flag=="9f2":
            R_b=parameters.amplitude
            
        self.nominal_amplitudes=np.ones(self.n_oscillators) 
        self.nominal_amplitudes[0:self.n_oscillators_body]=R_b
        self.nominal_amplitudes[self.n_oscillators_body:self.n_oscillators]=R_l
        print(self.nominal_amplitudes[20])
        
        if parameters.flag=="9c":
            self.nominal_amplitudes[0:int(self.n_oscillators_body/2)]=np.linspace(parameters.amplitude_gradient[0],parameters.amplitude_gradient[1],int(self.n_oscillators_body/2))
            self.nominal_amplitudes[int(self.n_oscillators_body/2):self.n_oscillators_body]=self.nominal_amplitudes[0:int(self.n_oscillators_body/2)]
        
        if parameters.turn !=0 and parameters.turning:
            d1=parameters.drive+parameters.turn
            d2=parameters.drive-parameters.turn
            R_b_1=(0.196+0.065*d1)
            R_b_2=(0.196+0.065*d2)
            if d1<1:
                R_b_1=0
            if d1>5:
                R_b_1=0
            if d2<1:
                R_b_2=0
            if d2>5:
                R_b_2=0      
            print(R_b_1,R_b_2,parameters.turn)
            self.nominal_amplitudes=np.ones(self.n_oscillators) 
            self.nominal_amplitudes[0:self.n_oscillators_body//2]=R_b_1
            self.nominal_amplitudes[self.n_oscillators_body//2:self.n_oscillators_body]=R_b_2
            self.nominal_amplitudes[self.n_oscillators_body:self.n_oscillators]=R_l