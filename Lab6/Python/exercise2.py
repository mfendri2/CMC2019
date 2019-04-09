""" Lab 6 Exercise 2

This file implements the pendulum system with two muscles attached

"""

from math import sqrt

import cmc_pylog as pylog
import numpy as np
from matplotlib import pyplot as plt

from cmcpack import DEFAULT
from cmcpack.plot import save_figure
from muscle import Muscle
from muscle_system import MuscleSytem
from neural_system import NeuralSystem
from pendulum_system import PendulumSystem
from system import System
from system_animation import SystemAnimation
from system_parameters import (MuscleParameters, NetworkParameters,
                               PendulumParameters)
from system_simulation import SystemSimulation


# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True
def exercise2a():
    m1_origin = np.array([-0.17, 0.0])  # Origin of Muscle 1
    m1_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 1

    m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 2

    a21=-0.10
    a11=-0.17
    
    a22=0.17
    a12=-0.17



    theta=np.arange(-np.pi/4,np.pi/4,0.01)
    length1=np.sqrt(np.power(a21,2)+np.power(a11,2)+2*a11*a21*np.sin(theta))
    length2=np.sqrt(np.power(a22,2)+np.power(a12,2)+2*a12*a22*np.sin(theta))
    h1=np.divide(a11*a21*np.cos(theta),length1)
    h2=np.divide(a12*a22*np.cos(theta),length2)
    plt.figure("Muscle Length as a Function of Theta")
    plt.plot(theta,length1)
    plt.plot(theta,length2)
    plt.xlabel("Theta [rad]")
    plt.ylabel("Muscle Length [m]")
    plt.title("Muscle Length")
    plt.show()
    
    plt.figure("h as funct of theta")
    plt.xlabel("Theta [rad]")
    plt.ylabel("Moment arm h [M]")
    plt.title("Moment Arm")
    plt.plot(theta,h1)
    plt.plot(theta,np.abs(h2))
    
def exercise2c():
    pendulum_params = PendulumParameters()  # Instantiate pendulum parameters
    pendulum_params.L = 0.5  # To change the default length of the pendulum
    pendulum_params.m = 1.  # To change the default mass of the pendulum
    pendulum = PendulumSystem(pendulum_params)  # Instantiate Pendulum object

    #### CHECK OUT PendulumSystem.py to ADD PERTURBATIONS TO THE MODEL #####

    pylog.info('Pendulum model initialized \n {}'.format(
        pendulum.parameters.showParameters()))

    # Define and Setup your pendulum model here
    # Check MuscleSytem.py for more details on MuscleSytem class
    M1_param = MuscleParameters()  # Instantiate Muscle 1 parameters
    M1_param.f_max = 1500  # To change Muscle 1 max force
    M2_param = MuscleParameters()  # Instantiate Muscle 2 parameters
    M2_param.f_max = 1500  # To change Muscle 2 max force
    M1 = Muscle(M1_param)  # Instantiate Muscle 1 object
    M2 = Muscle(M2_param)  # Instantiate Muscle 2 object
    # Use the MuscleSystem Class to define your muscles in the system
    muscles = MuscleSytem(M1, M2)  # Instantiate Muscle System with two muscles
    pylog.info('Muscle system initialized \n {} \n {}'.format(
        M1.parameters.showParameters(),
        M2.parameters.showParameters()))

    # Define Muscle Attachment points
    m1_origin = np.array([-0.17, 0.0])  # Origin of Muscle 1
    m1_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 1

    m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 2

    # Attach the muscles
    muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))

    # Create a system with Pendulum and Muscles using the System Class
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system

    ##### Model Initial Conditions #####
    x0_P = np.array([0, 0.])  # Pendulum initial condition

    # Muscle Model initial condition
    x0_M = np.array([0., M1.L_OPT, 0., M2.L_OPT])

    x0 = np.concatenate((x0_P, x0_M))  # System initial conditions

    ##### System Simulation #####
    # For more details on System Simulation check SystemSimulation.py
    # SystemSimulation is used to initialize the system and integrate
    # over time

    sim = SystemSimulation(sys)  # Instantiate Simulation object
    
    #Frequency effect :
    
    stim_frequency = np.array([0.05,0.1,0.5,1,5,10,50,100,500]) #in Hz
    
    stim_amplitude = 1 # belongs to 0-1
    phase = np.pi
    
    frequency_pendelum=np.zeros(len(stim_frequency))
    amplitude_pendelum=np.zeros(len(stim_frequency))
    for j,frequency in enumerate(stim_frequency):
        t_max = 5/frequency  # Maximum simulation time
        time_step = 0.001*(1/frequency)
        time = np.arange(0., t_max, time_step)  # Time vector

        act1 = np.zeros((len(time),1))
        act2 = np.zeros((len(time),1))
        act1[:,0] = stim_amplitude*(1 + np.sin(2*np.pi*frequency*time))/2
        act2[:,0] = stim_amplitude*(1+ np.sin(2*np.pi*frequency*time + phase))/2
        activations = np.hstack((act1, act2))
        sim.add_muscle_activations(activations)
        sim.initalize_system(x0, time)  # Initialize the system state
        sim.simulate()
        res = sim.results()  
        #computing the freuquency and amplitude
        angular_position = res[:,1]
        #signal_stat = signal[index_start:len(signal)]
        start_index=int(len(angular_position)/2)
        final_index=(len(angular_position))
        index_zeros = np.where(np.diff(np.sign(angular_position[start_index:final_index])))[0] #np.where(signal_stat==0)[0]
        deltas = np.diff(index_zeros)
        delta = np.mean(deltas)
       
        frequency_pendelum[j] = 1/(2*delta*time_step)
        signal = angular_position[start_index:len(angular_position)]
        amplitude = (np.max(signal)-np.min(signal))/2
        amplitude_pendelum[j] = amplitude
    
    
    plt.figure()
    plt.subplot(121)
    plt.loglog(stim_frequency,frequency_pendelum)
    plt.grid()
    plt.xlabel('Stimulation Frequency [Hz]')
    plt.ylabel('Pendulum Oscillation Frequency [Hz]')
    plt.subplot(122)
    plt.loglog(stim_frequency,amplitude_pendelum)
    plt.grid()
    plt.xlabel('Stimulation Frequency [Hz]')
    plt.ylabel('Pendulum Oscillation Amplitude [rad]')
    plt.savefig('2c.png')
    plt.show()
    stim_frequency = 10 #in Hz
    stim_amplitude = np.arange(0,1.1,0.1)
    frequency_pendelum=np.zeros(len(stim_amplitude))
    amplitude_pendelum=np.zeros(len(stim_amplitude))
    
    for j,amplitude_ in enumerate(stim_amplitude):
        t_max = 5/stim_frequency  # Maximum simulation time
        time_step = 0.001*(1/stim_frequency)
        time = np.arange(0., t_max, time_step)  # Time vector
    
        act1 = np.zeros((len(time),1))
        act2 = np.zeros((len(time),1))
        act1[:,0] = amplitude_*(1 + np.sin(2*np.pi*stim_frequency*time))/2
        act2[:,0] = amplitude_*(1+ np.sin(2*np.pi*stim_frequency*time + phase))/2
        activations = np.hstack((act1, act2))
        sim.add_muscle_activations(activations)
        sim.initalize_system(x0, time)  # Initialize the system state
        sim.simulate()
        res = sim.results()  
        #computing the freuquency and amplitude
        angular_position = res[:,1]
        #signal_stat = signal[index_start:len(signal)]
        start_index=int(len(angular_position)/2)
        final_index=(len(angular_position))
        index_zeros = np.where(np.diff(np.sign(angular_position[start_index:final_index])))[0] #np.where(signal_stat==0)[0]
        deltas = np.diff(index_zeros)
        delta = np.mean(deltas)
       
        frequency_pendelum[j] = 1/(2*delta*time_step)
        signal = angular_position[start_index:len(angular_position)]
        amplitude = (np.max(signal)-np.min(signal))/2
        amplitude_pendelum[j] = amplitude
    frequency_pendelum[0]=0
    plt.figure()
    plt.subplot(121)
    plt.plot(stim_amplitude,frequency_pendelum)
    plt.grid()
    plt.xlabel('Stimulation Amplitude [rad]')
    plt.ylabel('Pendulum Oscillation Frequency [Hz]')
    plt.subplot(122)
    plt.plot(stim_amplitude,amplitude_pendelum)
    plt.grid()
    plt.xlabel('Stimulation Amplitude[rad]')
    plt.ylabel('Pendulum Oscillation Amplitude [rad]')
    plt.savefig('2c_amplitude.png')
    plt.show()
def exercise2():
    """ Main function to run for Exercise 2.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """

    # Define and Setup your pendulum model here
    # Check PendulumSystem.py for more details on Pendulum class
    pendulum_params = PendulumParameters()  # Instantiate pendulum parameters
    pendulum_params.L = 0.5  # To change the default length of the pendulum
    pendulum_params.m = 1.  # To change the default mass of the pendulum
    pendulum = PendulumSystem(pendulum_params)  # Instantiate Pendulum object

    #### CHECK OUT PendulumSystem.py to ADD PERTURBATIONS TO THE MODEL #####

    pylog.info('Pendulum model initialized \n {}'.format(
        pendulum.parameters.showParameters()))

    # Define and Setup your pendulum model here
    # Check MuscleSytem.py for more details on MuscleSytem class
    M1_param = MuscleParameters()  # Instantiate Muscle 1 parameters
    M1_param.f_max = 1500  # To change Muscle 1 max force
    M2_param = MuscleParameters()  # Instantiate Muscle 2 parameters
    M2_param.f_max = 1500  # To change Muscle 2 max force
    M1 = Muscle(M1_param)  # Instantiate Muscle 1 object
    M2 = Muscle(M2_param)  # Instantiate Muscle 2 object
    # Use the MuscleSystem Class to define your muscles in the system
    muscles = MuscleSytem(M1, M2)  # Instantiate Muscle System with two muscles
    pylog.info('Muscle system initialized \n {} \n {}'.format(
        M1.parameters.showParameters(),
        M2.parameters.showParameters()))

    # Define Muscle Attachment points
    m1_origin = np.array([-0.17, 0.0])  # Origin of Muscle 1
    m1_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 1

    m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 2

    # Attach the muscles
    muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))

    # Create a system with Pendulum and Muscles using the System Class
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system

    ##### Time #####
    t_max = 2.5  # Maximum simulation time
    time = np.arange(0., t_max, 0.001)  # Time vector

    ##### Model Initial Conditions #####
    x0_P = np.array([np.pi/6, 0.])  # Pendulum initial condition

    # Muscle Model initial condition
    x0_M = np.array([0., M1.L_OPT, 0., M2.L_OPT])

    x0 = np.concatenate((x0_P, x0_M))  # System initial conditions

    ##### System Simulation #####
    # For more details on System Simulation check SystemSimulation.py
    # SystemSimulation is used to initialize the system and integrate
    # over time

    sim = SystemSimulation(sys)  # Instantiate Simulation object

    # Add muscle activations to the simulation
    # Here you can define your muscle activation vectors
    # that are time dependent
    sin_freq=1 #hz
    ampl_sin=1
    phase_difference_1_2=np.pi
    act1 = np.ones((len(time), 1)) 
    act2 = np.ones((len(time), 1)) 
    for i in range(len(time)):
        act1[i,0]=ampl_sin * (1+np.sin(2*np.pi*sin_freq*time[i]))
        act2[i,0]=ampl_sin * (1+np.sin(2*np.pi*sin_freq*time[i]+phase_difference_1_2))
    activations = np.hstack((act1, act2))

    # Method to add the muscle activations to the simulation

    sim.add_muscle_activations(activations)

    # Simulate the system for given time

    sim.initalize_system(x0, time)  # Initialize the system state

    #: If you would like to perturb the pedulum model then you could do
    # so by
    sim.sys.pendulum_sys.parameters.PERTURBATION = True
    # The above line sets the state of the pendulum model to zeros between
    # time interval 1.2 < t < 1.25. You can change this and the type of
    # perturbation in
    # pendulum_system.py::pendulum_system function

    # Integrate the system for the above initialized state and time
    sim.simulate()

    # Obtain the states of the system after integration
    # res is np.array [time, states]
    # states vector is in the same order as x0
    res = sim.results()

    # In order to obtain internal states of the muscle
    # you can access the results attribute in the muscle class
    muscle1_results = sim.sys.muscle_sys.Muscle1.results
    muscle2_results = sim.sys.muscle_sys.Muscle2.results

    # Plotting the results
    plt.figure('Pendulum')
    plt.title('Pendulum Phase')
    plt.plot(res[:, 1], res[:, 2])
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid()
    plt.figure('Activations')
    plt.title('Sine wave activations for both muscles')
    plt.plot(time,act1)
    plt.plot(time,act2)
    plt.legend(("activation muscle1","activation muscle2"))
    # To animate the model, use the SystemAnimation class
    # Pass the res(states) and systems you wish to animate
    simulation = SystemAnimation(res, pendulum, muscles)
    # To start the animation
    if DEFAULT["save_figures"] is False:
        simulation.animate()

    if not DEFAULT["save_figures"]:
        plt.show()
    else:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise2()

