"""CMC robot"""

import numpy as np
from network import SalamanderNetwork
from experiment_logger import ExperimentLogger
import math

class SalamanderCMC(object):
    """Salamander robot for CMC"""

    N_BODY_JOINTS = 10
    N_LEGS = 4

    def __init__(self, robot, n_iterations, parameters, logs="logs/log.npz"):
        super(SalamanderCMC, self).__init__()
        self.robot = robot
        timestep = int(robot.getBasicTimeStep())
        self.network = SalamanderNetwork(1e-3*timestep, parameters)

        # Position sensors
        self.position_sensors = [
            self.robot.getPositionSensor('position_sensor_{}'.format(i+1))
            for i in range(self.N_BODY_JOINTS)
        ] + [
            self.robot.getPositionSensor('position_sensor_leg_{}'.format(i+1))
            for i in range(self.N_LEGS)
        ]
        for sensor in self.position_sensors:
            sensor.enable(timestep)

        # GPS
        self.gps = robot.getGPS("fgirdle_gps")
        self.gps.enable(timestep)

        # Get motors
        self.motors_body = [
            self.robot.getMotor("motor_{}".format(i+1))
            for i in range(self.N_BODY_JOINTS)
        ]
        self.motors_legs = [
            self.robot.getMotor("motor_leg_{}".format(i+1))
            for i in range(self.N_LEGS)
        ]

        # Set motors
        for motor in self.motors_body:
            motor.setPosition(0)
            motor.enableForceFeedback(timestep)
            motor.enableTorqueFeedback(timestep)
        for motor in self.motors_legs:
            motor.setPosition(-np.pi/2)
            motor.enableForceFeedback(timestep)
            motor.enableTorqueFeedback(timestep)

        # Iteration counter
        self.iteration = 0

        # Logging
        self.log = ExperimentLogger(
            n_iterations,
            n_links=1,
            n_joints=self.N_BODY_JOINTS+self.N_LEGS,
            filename=logs,
            timestep=1e-3*timestep,
            **parameters
        )
    def get_coord(self):
        return self.gps.getValues()

    def log_iteration(self):
        """Log state"""
        self.log.log_link_positions(self.iteration, 0, self.gps.getValues())
        for i, motor in enumerate(self.motors_body):
            # Position
            self.log.log_joint_position(
                self.iteration, i,
                self.position_sensors[i].getValue()
            )
            # Command
            self.log.log_joint_cmd(
                self.iteration, i,
                motor.getTargetPosition()
            )
            # Torque
            self.log.log_joint_torque(
                self.iteration, i,
                motor.getTorqueFeedback()
            )
            # Torque feedback
            self.log.log_joint_torque_feedback(
                self.iteration, i,
                motor.getTorqueFeedback()
            )
        for i, motor in enumerate(self.motors_legs):
            # Position
            self.log.log_joint_position(
                self.iteration, 10+i,
                self.position_sensors[10+i].getValue()
            )
            # Command
            self.log.log_joint_cmd(
                self.iteration, 10+i,
                motor.getTargetPosition()
            )
            # Torque
            self.log.log_joint_torque(
                self.iteration, 10+i,
                motor.getTorqueFeedback()
            )
            # Torque feedback
            self.log.log_joint_torque_feedback(
                self.iteration, 10+i,
                motor.getTorqueFeedback()
            )

    def step(self,parameters):
        """Step"""
        # Increment iteration
        self.iteration += 1
        # Update network
        self.network.step()
        positions = self.network.get_motor_position_output()
        if parameters.flag == "9g" and self.gps.getValues()[0]>0.4 and parameters.toggle=="walk":
            parameters.toggle="swim"
            parameters.drive=4
            print("d={}".format(parameters.drive))
            self.network.parameters.update(parameters)
        if parameters.flag == "9g" and self.gps.getValues()[0]<0.2 and parameters.toggle=="swim":
            parameters.drive=2
            print("d={}".format(parameters.drive))
            self.network.parameters.update(parameters)
            parameters.toggle="walk"
            self.network.reset_leg_phases()
            
        if parameters.flag=="9d1" and self.iteration==2000:
            print("turn on")
            parameters.turning=True
            self.network.parameters.update(parameters)
        if parameters.flag=="9d1" and self.iteration==4000:
            print("turn off")
            parameters.turning=False
            self.network.parameters.update(parameters)
        
        
        
        # Update control
        for i in range(self.N_BODY_JOINTS):
            self.motors_body[i].setPosition(positions[i])
        
        if np.all(abs(self.network.parameters.nominal_amplitudes[20:24])<=0.0001):
            for i in range(self.N_LEGS):
                self.motors_legs[i].setPosition(
                    round(self.position_sensors[i+10].getValue()/(2*math.pi))*2*math.pi-math.pi/2
                )
        else:  
            for i in range(self.N_LEGS):
                self.motors_legs[i].setPosition(
                    positions[self.N_BODY_JOINTS+i] - np.pi/2
                    
                )

        # Log data
        self.log_iteration()

