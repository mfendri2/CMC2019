""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
import numpy as np

import cmc_pylog as pylog
from muscle import Muscle
from mass import Mass
from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from system_parameters import MuscleParameters, MassParameters
from isometric_muscle_system import IsometricMuscleSystem
from isotonic_muscle_system import IsotonicMuscleSystem

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True


def exercise1a():
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles
    parameters = MuscleParameters()
    pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())
    pylog.info("Use the parameters object to change the muscle parameters")

    # Create muscle object
    muscle = Muscle(parameters)

    pylog.warning("Isometric muscle contraction to be completed")
    
    
    
    # Instatiate isometric muscle system
    sys = IsometricMuscleSystem()

    # Add the muscle to the system
    
    sys.add_muscle(muscle)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Evalute for a single muscle stretch
    muscle_stretch = 0.2
    
    # Evalute for a single muscle stimulation
    muscle_stimulation = 1.

    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT]
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contracticle length initial value

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.2
    time_step = 0.001

    time = np.arange(t_start, t_stop, time_step)

    # Run the integration
    result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           stimulation=muscle_stimulation,
                           muscle_length=muscle_stretch)
    
    

    # 
    muscle_length=np.arange(0,0.4,0.001)
#    F_active=[] 
#    F_passive=[] 
#    F_total=[] 
#    F_length=[]
#    # Exercice 1 a
#    for length in muscle_length:
#        result = sys.integrate(x0=x0,
#                           time=time,
#                           time_step=time_step,
#                           stimulation=muscle_stimulation,
#                           muscle_length=length)
#        F_active.append(result.active_force[-1])
#        F_passive.append(result.passive_force[-1])  
#        F_total.append(result.active_force[-1]+result.passive_force[-1])
#        F_length.append(result.l_ce[-1]) 
#  
#     
#    plt.figure('Isometric muscle experiment')
#   
#    plt.plot(F_length,F_active)
#    plt.plot(F_length,F_passive)
#    plt.plot(F_length,F_total)
#    plt.title('Isometric muscle experiment')
#    plt.xlabel('Length [m]')
#    plt.ylabel('Muscle force [N]')
#    plt.legend(("Active","Passive","Total force"))
#
#    plt.grid()
#    plt.show()
#    # 
#    plt.figure("Isometric muscle experiment by changing the stimulation")
#    different_stimulation=np.arange(0,1,0.2) 
##    for stimulation in different_stimulation:
#        F_total=[] 
#        F_length=[]
#        for length in muscle_length: 
#            result = sys.integrate(x0=x0,
#                           time=time,
#                           time_step=time_step,
#                           stimulation=stimulation,
#                           muscle_length=length)
#            F_total.append(result.active_force[-1]+result.passive_force[-1])
#            F_length.append(result.l_ce[-1]) 
#        plt.plot(F_length,F_total)
#    
#    plt.xlabel('Length [m]')
#    plt.ylabel('Total muscle force [N]')
#    plt.legend(("stimulation = 0","stimulation = 0.2","stimulation = 0.4","stimulation = 0.6","stimulation = 0.8","stimulation = 1"))       
#    plt.grid()
#    plt.show()
    
    # 1/c 
    fiber_opt_small=0.07 
    fiber_opt_medium=0.11
    fiber_opt_long=0.16 
    lopt_list=[fiber_opt_small,fiber_opt_medium,fiber_opt_long]
    muscle_stimulation = 1
    plt.figure("Isometric muscle experiment changin optimal length")
    for lopt in lopt_list:
        print("RUNNING lopt=",lopt)
        parameters = MuscleParameters(l_opt=lopt)
        muscle = Muscle(parameters)
        sys = IsometricMuscleSystem()
        sys.add_muscle(muscle)
        F_total=[] 
        F_length=[]
        for length in muscle_length: 
            result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           stimulation=muscle_stimulation,
                           muscle_length=length)
            F_total.append(result.active_force[-1]+result.passive_force[-1])
            F_length.append(result.l_ce[-2]) 
        plt.plot(F_length,F_total)
    plt.xlabel('Length [m]')
    plt.ylabel('Total muscle force [N]')
    plt.legend(("Small l_opt","Medium l_opt","Long l_opt"))       
    plt.grid()
    plt.show()
    
    
def exercise1d():
    """ Exercise 1d

    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest."""

    # Defination of muscles
    muscle_parameters = MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters = MassParameters()
    print(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)

    pylog.warning("Isotonic muscle contraction to be implemented")

    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Evalute for a single load
    load = 100.

    # Evalute for a single muscle stimulation
    muscle_stimulation = 1.

    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT,
          sys.muscle.L_OPT + sys.muscle.L_SLACK, 0.0]
    # x0[0] - -> activation
    # x0[1] - -> contractile length(l_ce)
    # x0[2] - -> position of the mass/load
    # x0[3] - -> velocity of the mass/load

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.3
    time_step = 0.001
    time_stabilize = 0.2

    time = np.arange(t_start, t_stop, time_step)

    # Run the integration
    load_array=np.arange(0,25,1)
    vel_ce=[]
    for load_ in load_array:
        
        result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           time_stabilize=time_stabilize,
                           stimulation=muscle_stimulation,
                           load=load_)
        
        if result.l_mtc[-1] > ( muscle_parameters.l_opt + muscle_parameters.l_slack): 
            
            vel_ce.append(min(result.v_ce[:]))
        else: 
            vel_ce.append(max(result.v_ce[:]))
                
    
    plt.figure('Isotonic muscle experiment')
    plt.plot(vel_ce,load_array*mass_parameters.g)
    plt.title('Isotonic muscle experiment')
    plt.xlabel('Contractile element velocity [m/s]')
    plt.ylabel('External load applied [N]')
    plt.grid()


def exercise1():
    #exercise1a()
    exercise1d()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        print(figures)
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise1()

