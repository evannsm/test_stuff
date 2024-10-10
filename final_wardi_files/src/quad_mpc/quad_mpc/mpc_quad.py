# TODO: 1) change time.time() to time.process_time


wardiNN_on = bool(int(input("Is the wardiNN conda env activated? Press 0 for No and 1 for Yes: ")))
if wardiNN_on:
    print("You're all set :)")
else:
    print("Sorry you need to set up the conda environment and have it activated")
    exit(1)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleRatesSetpoint, VehicleCommand, VehicleStatus, VehicleOdometry, TrajectorySetpoint, RcChannels
from std_msgs.msg import Float64MultiArray

from .workingModel import Quadrotor
from .workingGenMPC import QuadrotorMPC2

# from tf_transformations import euler_from_quaternion
import numpy as np
import math as m
import time

from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain
from pyJoules.energy_meter import EnergyContext


import sys
import traceback
from .Logger import Logger


class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""
    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

###############################################################################################################################################

        # Figure out if in simulation or hardware mode to set important variables to the appropriate values
        self.sim = bool(int(input("Are you using the simulator? Write 1 for Sim and 0 for Hardware: ")))
        print(f"{'SIMULATION' if self.sim else 'HARDWARE'}")

        self.ctrl_loop_time_log = []
        self.x_log, self.y_log, self.z_log, self.yaw_log = [], [], [], []
        self.throttle_log, self.roll_log, self.pitch_log, self.yaw_rate_log = [], [], [], []
        self.ref_x_log, self.ref_y_log, self.ref_z_log, self.ref_yaw_log = [], [], [], []
        self.mpc_timel_array = []


        self.pyjoules_on = False
        if self.pyjoules_on:
            self.csv_handler = CSVHandler('mpc_cpu_energy_TESTERRORS.csv')
###############################################################################################################################################

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )


        # Create Publishers
        # Publishers for Setting to Offboard Mode and Arming/Diasarming/Landing/etc
        self.offboard_control_mode_publisher = self.create_publisher( #publishes offboard control heartbeat
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher( #publishes vehicle commands (arm, offboard, disarm, etc)
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        
        # Publishers for Sending Setpoints in Offboard Mode: 1) Body Rates and Thrust, 2) Position and Yaw 
        self.rates_setpoint_publisher = self.create_publisher( #publishes body rates and thrust setpoint
            VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher( #publishes trajectory setpoint
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        
        # Publisher for Logging States, Inputs, and Reference Trajectories for Data Analysis
        self.state_input_ref_log_publisher_ = self.create_publisher( #publishes log of states and input
            Float64MultiArray, '/state_input_ref_log', 10)
        self.state_input_ref_log_msg = Float64MultiArray() #creates message for log of states and input        

        # Create subscribers
        self.vehicle_odometry_subscriber = self.create_subscription( #subscribes to odometry data (position, velocity, attitude)
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription( #subscribes to vehicle status (arm, offboard, disarm, etc)
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
    
        self.offboard_mode_rc_switch_on = True if self.sim else False #Offboard mode starts on if in Sim, turn off and wait for RC if in hardware
        self.rc_channels_subscriber = self.create_subscription( #subscribes to rc_channels topic for software "killswitch" to make sure we'd like position vs offboard vs land mode
            RcChannels, '/fmu/out/rc_channels', self.rc_channel_callback, qos_profile
        )

###############################################################################################################################################

        # Initialize variables:  
        self.time_before_land = 10.0
        print(f"time_before_land: {self.time_before_land}")
        self.offboard_setpoint_counter = 0 #helps us count 10 cycles of sending offboard heartbeat before switching to offboard mode and arming
        self.vehicle_status = VehicleStatus() #vehicle status variable to make sure we're in offboard mode before sending setpoints

        self.T0 = time.time() # initial time of program
        self.time_from_start = time.time() - self.T0 # time from start of program initialized and updated later to keep track of current time in program
        self.first_iteration = True #boolean to help us initialize the first iteration of the program
###############################################################################################################################################
        
        if self.sim:
            print("Using simulator throttle from force conversion function")
            self.MASS = 1.535 #set simulation mass from iris model sdf for linearized model calculations
            # The following 3 variables are used to convert between force and throttle commands for the iris drone defined in PX4 stack for gazebo simulation
            self.MOTOR_CONSTANT = 0.00000584 #iris gazebo simulation motor constant
            self.MOTOR_VELOCITY_ARMED = 100 #iris gazebo motor velocity when armed
            self.MOTOR_INPUT_SCALING = 1000.0 #iris gazebo simulation motor input scaling
        elif not self.sim:
            print("Using hardware throttle from force conversion function and certain trajectories will not be available")
            self.MASS = 1.75 #alternate: 1.69kg and have grav_mass = 2.10

        self.GRAVITY = 9.806 #gravity
        self.T_lookahead = 0. #lookahead time for prediction and reference tracking in controller

###############################################################################################################################################
        # Load Up MPC controller from the Imported Classes
        quad = Quadrotor(sim=self.sim)
        generate_c_code = False
        horizon = 3.0 #2, 3
        num_steps = 20 #10, 20
        self.mpc_solver = QuadrotorMPC2(generate_c_code, quad, horizon, num_steps)
        self.num_steps = num_steps


        self.metadata = np.array(['Sim' if self.sim else 'Hardware',
                                  'Horizon:'+str(horizon),
                                  'Num Steps:'+str(num_steps),
                                  'Pyjoules' if self.pyjoules_on else 'No Pyjoules',
                                  ])
###############################################################################################################################################

        #Create Function @ {1/self.offboard_timer_period}Hz (in my case should be 10Hz/0.1 period) to Publish Offboard Control Heartbeat Signal
        self.offboard_timer_period = 0.1
        self.timer = self.create_timer(self.offboard_timer_period, self.offboard_mode_timer_callback)
        # exit(0)

        # Create Function at {1/self.controller_timer_period}Hz (in my case should be 100Hz/0.01 period) to Send Control Input
        self.controller_timer_period = 0.01
        self.timer = self.create_timer(self.controller_timer_period, self.controller_timer_callback)

    # The following 4 functions all call publish_vehicle_command to arm/disarm/land/ and switch to offboard mode
    # The 5th function publishes the vehicle command
    # The 6th function checks if we're in offboard mode
    # The 7th function handles the safety RC control switches for hardware
    def arm(self): #1. Sends arm command to vehicle via publish_vehicle_command function
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self): #2. Sends disarm command to vehicle via publish_vehicle_command function
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self): #3. Sends offboard command to vehicle via publish_vehicle_command function
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self): #4. Sends land command to vehicle via publish_vehicle_command function
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_vehicle_command(self, command, **params) -> None: #5. Called by the above 4 functions to send parameter/mode commands to the vehicle
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def vehicle_status_callback(self, vehicle_status): #6. This function helps us check if we're in offboard mode before we start sending setpoints
        """Callback function for vehicle_status topic subscriber."""
        # print('vehicle status callback')
        self.vehicle_status = vehicle_status

    def rc_channel_callback(self, rc_channels):
        """Callback function for RC Channels to create a software 'killswitch' depending on our flight mode channel (position vs offboard vs land mode)"""
        print('rc channel callback')
        self.mode_channel = 5
        flight_mode = rc_channels.channels[self.mode_channel-1] # +1 is offboard everything else is not offboard
        self.offboard_mode_rc_switch_on = True if flight_mode >= 0.75 else False

    # The following 2 functions are used to publish offboard control heartbeat signals
    def publish_offboard_control_heartbeat_signal2(self): #1)Offboard Signal2 for Returning to Origin with Position Control
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_offboard_control_heartbeat_signal1(self): #2)Offboard Signal1 for Newton-Rapshon Body Rate Control
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)


# ~~ The remaining functions are all intimately related to the MPC Control Algorithm ~~
    # The following 2 functions are used to convert between force and throttle commands
    def get_throttle_command_from_force(self, collective_thrust): #Converts force to throttle command
        """ Convert the collective thrust to a throttle command. """
        collective_thrust = -collective_thrust
        # print(f"Conv2Throttle: collective_thrust: {collective_thrust}")
        if self.sim:
            motor_speed = m.sqrt(collective_thrust / (4.0 * self.MOTOR_CONSTANT))
            throttle_command = (motor_speed - self.MOTOR_VELOCITY_ARMED) / self.MOTOR_INPUT_SCALING
            return -throttle_command
        
        if not self.sim:
            a = 0.00705385408507030
            b = 0.0807474474438391
            c = 0.0252575818743285

            # equation form is a*x + b*sqrt(x) + c = y
            throttle_command = a*collective_thrust + b*m.sqrt(collective_thrust) + c
            return -throttle_command

    def get_force_from_throttle_command(self, throttle_command): #Converts throttle command to force
        """ Convert the throttle command to a collective thrust. """
        throttle_command = -throttle_command
        print(f"Conv2Force: throttle_command: {throttle_command}")
        if self.sim:
            motor_speed = (throttle_command * self.MOTOR_INPUT_SCALING) + self.MOTOR_VELOCITY_ARMED
            collective_thrust = 4.0 * self.MOTOR_CONSTANT * motor_speed ** 2
            return -collective_thrust
        
        if not self.sim:
            a = 19.2463167420814
            b = 41.8467162352942
            c = -7.19353022443441

            # equation form is a*x^2 + b*x + c = y
            collective_thrust = a*throttle_command**2 + b*throttle_command + c
            return -collective_thrust

    def euler_from_matrix(self, matrix, axes='sxyz'):
        """Return Euler angles from rotation matrix for specified axis sequence.

        axes : One of 24 axis sequences as string or encoded tuple

        Note that many Euler angle triplets can describe one matrix.

        # >>> R0 = euler_matrix(1, 2, 3, 'syxz')
        # >>> al, be, ga = euler_from_matrix(R0, 'syxz')
        # >>> R1 = euler_matrix(al, be, ga, 'syxz')
        # >>> numpy.allclose(R0, R1)
        # True
        # >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
        # >>> for axes in _AXES2TUPLE.keys():
        # ...    R0 = euler_matrix(axes=axes, *angles)
        # ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
        # ...    if not numpy.allclose(R0, R1): print axes, "failed"

        # """

        # epsilon for testing whether a number is close to zero
        _EPS = np.finfo(float).eps * 4.0

        # axis sequences for Euler angles
        _NEXT_AXIS = [1, 2, 0, 1]

        # map axes strings to/from tuples of inner axis, parity, repetition, frame
        _AXES2TUPLE = {
            'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
            'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
            'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
            'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
            'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
            'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
            'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
            'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

        _TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

        try:
            firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
        except (AttributeError, KeyError):
            _ = _TUPLE2AXES[axes]
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = _NEXT_AXIS[i+parity]
        k = _NEXT_AXIS[i-parity+1]

        M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
        if repetition:
            sy = m.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
            if sy > _EPS:
                ax = m.atan2( M[i, j],  M[i, k])
                ay = m.atan2( sy,       M[i, i])
                az = m.atan2( M[j, i], -M[k, i])
            else:
                ax = m.atan2(-M[j, k],  M[j, j])
                ay = m.atan2( sy,       M[i, i])
                az = 0.0
        else:
            cy = m.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
            if cy > _EPS:
                ax = m.atan2( M[k, j],  M[k, k])
                ay = m.atan2(-M[k, i],  cy)
                az = m.atan2( M[j, i],  M[i, i])
            else:
                ax = m.atan2(-M[j, k],  M[j, j])
                ay = m.atan2(-M[k, i],  cy)
                az = 0.0

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax
        return ax, ay, az

    def quaternion_matrix(self, quaternion):
        """Return homogeneous rotation matrix from quaternion.

        >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
        >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
        True

        """
        _EPS = np.finfo(float).eps * 4.0

        q = np.array(quaternion[:4], dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < _EPS:
            return np.identity(4)
        q *= m.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array((
            (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
            (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
            (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
            (                0.0,                 0.0,                 0.0, 1.0)
            ), dtype=np.float64)

    def euler_from_quaternion(self, quaternion, axes='sxyz'):
        """Return Euler angles from quaternion for specified axis sequence.

        >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
        >>> numpy.allclose(angles, [0.123, 0, 0])
        True

        """
        return self.euler_from_matrix(self.quaternion_matrix(quaternion), axes)

    def normalize_angle(self, angle):
        """ Normalize the angle to the range [-pi, pi]. """
        result = m.atan2(m.sin(angle), m.cos(angle))
        # print(f"normalize_angle: input={angle}, result={result}")
        return result

    def new_angle_wrapper(self, angle):
        """Wrap the angle to the range [-pi, pi]."""
        # angle_adj = angle + m.pi
        # normalized = self.normalize_angle(angle_adj)
        # result = -1 * normalized
        # print(f"new_angle_wrapper: input={angle}, normalized={normalized}, result={result}")
        return -1 * self.normalize_angle(angle + m.pi)

    def vehicle_odometry_callback(self, msg): # Odometry Callback Function Yields Position, Velocity, and Attitude Data
        """Callback function for vehicle_odometry topic subscriber."""
        # print("AT ODOM CALLBACK")
        (self.yaw, self.pitch, self.roll) = self.euler_from_quaternion(msg.q)
        # print("not yaw:")
        # not_yaw = self.new_angle_wrapper(self.yaw)
        # print("\nyaw: ")
        # self.yaw = self.old_angle_wrapper(self.yaw)

        self.yaw = self.new_angle_wrapper(self.yaw)
        self.pitch = -1 * self.pitch # pitch is negative of the value in gazebo bc of frame difference

        self.p = msg.angular_velocity[0]
        self.q = msg.angular_velocity[1]
        self.r = msg.angular_velocity[2]

        self.x = msg.position[0]
        self.y = msg.position[1]
        self.z = 1 * msg.position[2] # z is negative of the value in gazebo bc of frame difference

        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = 1 * msg.velocity[2] # vz is negative of the value in gazebo bc of frame difference

        # print(f"Roll: {self.roll}")
        # print(f"Pitch: {self.pitch}")
        # print(f"Yaw: {self.yaw}")
        
        self.stateVector = np.array([[self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw]]).T 
        self.nr_state = np.array([[self.x, self.y, self.z, self.yaw]]).T
        # self.odom_rates = np.array([[self.p, self.q, self.r]]).T

    def publish_rates_setpoint(self, thrust: float, roll: float, pitch: float, yaw: float): #Publishes Body Rate and Thrust Setpoints
        """Publish the trajectory setpoint."""
        msg = VehicleRatesSetpoint()
        msg.roll = float(roll)
        msg.pitch = float(pitch)
        msg.yaw = float(yaw)
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = 1* float(thrust)

        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.rates_setpoint_publisher.publish(msg)
        
        # print("in publish rates setpoint")
        # self.get_logger().info(f"Publishing rates setpoints [r,p,y]: {[roll, pitch, yaw]}")
        print(f"Publishing rates setpoints [thrust, r,p,y]: {[thrust, roll, pitch, yaw]}")

    def publish_position_setpoint(self, x: float, y: float, z: float): #Publishes Position and Yaw Setpoints
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 0.0  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")


# ~~ The following 2 functions are the main functions that run at 10Hz and 100Hz ~~
    def offboard_mode_timer_callback(self) -> None: # ~~Runs at 10Hz and Sets Vehicle to Offboard Mode  ~~
        """Offboard Callback Function for The 10Hz Timer."""
        # print("In offboard timer callback")

        if self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard deciding whether to send heartbeat signal, engage offboard, and arm
            if self.time_from_start <= self.time_before_land:
                self.publish_offboard_control_heartbeat_signal1()
            elif self.time_from_start > self.time_before_land:
                self.publish_offboard_control_heartbeat_signal2()


            if self.offboard_setpoint_counter == 10:
                self.engage_offboard_mode()
                self.arm()
            if self.offboard_setpoint_counter < 11:
                self.offboard_setpoint_counter += 1

        else:
            print(f"Offboard Callback: RC Flight Mode Channel {self.mode_channel} Switch Not Set to Offboard (-1: position, 0: offboard, 1: land) ")
            self.offboard_setpoint_counter = 0



    def controller_timer_callback(self) -> None: # ~~This is the main function that runs at 100Hz and Administrates Calls to Every Other Function ~~
        print("Controller Callback")
        if self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard deciding whether to send heartbeat signal, engage offboard, and arm
            # self.time_from_start = time.time()-self.T0 #update curent time from start of program for reference trajectories and for switching between my controller and landing mode
            
            print(f"--------------------------------------")
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                print("IN OFFBOARD MODE")
                print(f"Controller callback- timefromstart: {self.time_from_start}")
                
                if self.time_from_start <= self.time_before_land: # our controller for first {self.time_before_land} seconds
                    print(f"Entering MPC Control Loop for next: {self.time_before_land-self.time_from_start} seconds")
                    self.controller()

                elif self.time_from_start > self.time_before_land: #then land at origin and disarm
                    print("BACK TO SPAWN")
                    self.publish_position_setpoint(0.0, 0.0, -0.3)
                    print(f"self.x: {self.x}, self.y: {self.y}, self.z: {self.z}")
                    if abs(self.x) < 0.1 and abs(self.y) < 0.1 and abs(self.z) <= 0.50:
                        print("Switching to Land Mode")
                        self.land()

            if self.time_from_start > self.time_before_land:
                if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                        print("IN LAND MODE")
                        if abs(self.z) <= .24:
                            print("\nDisarming and Exiting Program")
                            self.disarm()
                            print("\nSaving all data!")
                            if self.pyjoules_on:
                                self.csv_handler.save_data()
                            exit(0)
            print(f"--------------------------------------")
            print("\n\n")
        else:
            print("Controller Callback: Channel 11 Switch Not Set to Offboard")

    
# ~~ From here down are the functions that actually calculate the control input ~~
    def controller(self): # Runs Algorithm Structure
        print(f"NR States: {self.nr_state}") #prints current state

        if self.first_iteration:
            print("First Iteration")
            self.T0 = time.time()
            self.first_iteration = False

        self.time_from_start = time.time()-self.T0


#~~~~~~~~~~~~~~~ Calculate reference trajectory ~~~~~~~~~~~~~~~
        if self.time_from_start <= 10.0:
            reffunc = self.hover_ref_func(1)
        else:
            reffunc = self.circle_horz_ref_func()
            reffunc = self.circle_horz_spin_ref_func()
            reffunc = self.circle_vert_ref_func()
            reffunc = self.fig8_horz_ref_func()
            reffunc = self.fig8_vert_ref_func_short()
            reffunc = self.fig8_vert_ref_func_tall()
            reffunc = self.helix()
            reffunc = self.helix_spin

        reffunc = self.hover_ref_func(1)
        print(f"reffunc: {reffunc}")

        # Calculate the MPC control input and transform the force into a throttle command for publishing to the vehicle
        new_u = self.get_new_control_input(reffunc)
        new_force = -new_u[0]       
        print(f"new_force: {new_force}")

        new_throttle = self.get_throttle_command_from_force(new_force)
        new_roll_rate = new_u[1]
        new_pitch_rate = new_u[2]
        new_yaw_rate = new_u[3]


        # Build the final input vector to save as self.u0 and publish to the vehicle via publish_rates_setpoint:
        final = [new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate]

        current_input_save = np.array(final).reshape(-1, 1)
        print(f"newInput: \n{current_input_save}")
        self.u0 = current_input_save

        # Publish the final input to the vehicle
        self.publish_rates_setpoint(final[0], final[1], final[2], final[3])

        # Log the states, inputs, and reference trajectories for data analysis
        state_input_ref_log_info = [float(self.x), float(self.y), float(self.z), float(self.yaw), float(final[0]), float(final[1]), float(final[2]), float(final[3]), float(reffunc[0][0]), float(reffunc[1][0]), float(reffunc[2][0]), float(reffunc[3][0]), self.time_from_start]
        self.update_logged_data(state_input_ref_log_info)
        # self.state_input_ref_log_msg.data = [float(self.x), float(self.y), float(self.z), float(self.yaw), float(final[0]), float(final[1]), float(final[2]), float(final[3]), float(reffunc[0][0]), float(reffunc[1][0]), float(reffunc[2][0]), float(reffunc[3][0])]
        # self.state_input_ref_log_publisher_.publish(self.state_input_ref_log_msg)

# ~~ The following functions handle the log update and data retrieval for analysis ~~
    def update_logged_data(self, data):
        # print("Updating Logged Data")
        self.x_log.append(data[0])
        self.y_log.append(data[1])
        self.z_log.append(data[2])
        self.yaw_log.append(data[3])
        self.throttle_log.append(data[4])
        self.roll_log.append(data[5])
        self.pitch_log.append(data[6])
        self.yaw_rate_log.append(data[7])
        self.ref_x_log.append(data[8])
        self.ref_y_log.append(data[9])
        self.ref_z_log.append(data[10])
        self.ref_yaw_log.append(data[11])
        self.ctrl_loop_time_log.append(data[12])
    def get_x_log(self): return np.array(self.x_log).reshape(-1, 1)
    def get_y_log(self): return np.array(self.y_log).reshape(-1, 1)
    def get_z_log(self): return np.array(self.z_log).reshape(-1, 1)
    def get_yaw_log(self): return np.array(self.yaw_log).reshape(-1, 1)
    def get_throttle_log(self): return np.array(self.throttle_log).reshape(-1, 1)
    def get_roll_log(self): return np.array(self.roll_log).reshape(-1, 1)
    def get_pitch_log(self): return np.array(self.pitch_log).reshape(-1, 1)
    def get_yaw_rate_log(self): return np.array(self.yaw_rate_log).reshape(-1, 1)
    def get_ref_x_log(self): return np.array(self.ref_x_log).reshape(-1, 1)
    def get_ref_y_log(self): return np.array(self.ref_y_log).reshape(-1, 1)
    def get_ref_z_log(self): return np.array(self.ref_z_log).reshape(-1, 1)
    def get_ref_yaw_log(self): return np.array(self.ref_yaw_log).reshape(-1, 1)
    def get_ctrl_loop_time_log(self): return np.array(self.ctrl_loop_time_log).reshape(-1, 1)
    def get_mpc_timel_log(self): return np.array(self.mpc_timel_array).reshape(-1, 1)
    def get_metadata(self): return self.metadata.reshape(-1, 1)

    def get_new_control_input(self, reffunc):
        if self.pyjoules_on:
            with EnergyContext(handler=self.csv_handler, domains=[RaplPackageDomain(0), RaplCoreDomain(0)]):
                return self.execute_control_input(reffunc)
        else:
            return self.execute_control_input(reffunc)
        
    def execute_control_input(self, reffunc):
        t0 = time.time()
        status, x_mpc, u_mpc = self.mpc_solver.solve_mpc_control(self.stateVector.flatten(), reffunc)
        mpc_timel = time.time() - t0
        print(f"Outside Acados Timel: {mpc_timel}sec, Good Enough for {1/mpc_timel}Hz")
        self.mpc_timel_array.append(mpc_timel)
        # print(f"status: {status}")
        # print(f"x_mpc: {x_mpc}")
        print(f"u_mpc: {u_mpc}")
        return u_mpc


# ~~ The following functions are reference trajectories for tracking ~~
    def hover_ref_func(self, num): #Returns Constant Hover Reference Trajectories At A Few Different Positions for Testing ([x,y,z,yaw])
        """ Returns constant hover reference trajectories at a few different positions. """
        hover_dict = {
            1: np.array([[0.0, 0.0, -0.6,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]).T,
            2: np.array([[0.0, 0.8, -0.8,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]).T,
            3: np.array([[0.8, 0.0, -0.8,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]).T,
            4: np.array([[0.8, 0.8, -0.8,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]).T,
            5: np.array([[0.0, 0.0, -10.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]).T,
            6: np.array([[1.0, 1.0, -4.0,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]).T,
            7: np.array([[3.0, 4.0, -5.0,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]).T,
            8: np.array([[1.0, 1.0, -3.0,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]).T,
        }

        if num > len(hover_dict) or num < 1:
            print(f"hover_dict #{num} not found")
            exit(0)

        if not self.sim:
            if num > 4:
                print("hover modes 5+ not available for hardware")
                exit(1)
            
        print(f"hover_dict #{num}")
        r = hover_dict.get(num)
        r_final = np.tile(r, (1, self.num_steps))
        return r_final
    
    def circle_horz_ref_func(self): #Returns Circle Reference Trajectory in Horizontal Plane ([x,y,z,yaw])
        """ Returns circle reference trajectory in horizontal plane. """
        print("circle_horz_ref_func")

        t = self.time_from_start + self.T_lookahead
        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637
        w = 2*m.pi / PERIOD

        x = .6*m.cos(w*t)
        y = .6*m.sin(w*t)
        z = -0.8
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll = 0.0
        pitch = 0.0
        yaw = 0.0

        r = np.array([[x, y, z, vx, vy, vz, roll, pitch, yaw]]).T
        r_final = np.tile(r, (1, self.num_steps))

        return r_final
    

    def circle_horz_spin_ref_func(self): #Returns Circle Reference Trajectory in Horizontal Plane while Yawing([x,y,z,yaw])
        """ Returns circle reference trajectory in horizontal plane while yawing. """
        print("circle_horz_spin_ref_func")

        t = self.time_from_start + self.T_lookahead
        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637
        w = 2*m.pi / PERIOD

        SPIN_PERIOD = 15

        x = .6*m.cos(w*t)
        y = .6*m.sin(w*t)
        z = -0.8
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll = 0.0
        pitch = 0.0
        yaw = t / (SPIN_PERIOD / (2*m.pi))

        r = np.array([[x, y, z, vx, vy, vz, roll, pitch, yaw]]).T
        r_final = np.tile(r, (1, self.num_steps))
        return r_final
    
    def circle_vert_ref_func(self): #Returns Circle Reference Trajectory in Vertical Plane ([x,y,z,yaw])
        """ Returns circle reference trajectory in horizontal plane. """        
        print("circle_vert_ref_func")

        t = self.time_from_start + self.T_lookahead
        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637
        w = 2*m.pi / PERIOD

        x = 0.0
        y = .35*m.cos(w*t)
        z = -1*( .35*m.sin(w*t) + .75 )
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll = 0.0
        pitch = 0.0
        yaw = 0.0

        r = np.array([[x, y, z, vx, vy, vz, roll, pitch, yaw]]).T
        r_final = np.tile(r, (1, self.num_steps))
        return r_final
    

    
    def fig8_horz_ref_func(self): #Returns Figure 8 Reference Trajectory in Horizontal Plane ([x,y,z,yaw])
        """ Returns figure 8 reference trajectory in horizontal plane. """
        print("fig8_horz_ref_func")

        t = self.time_from_start + self.T_lookahead
        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637
        w = 2*m.pi / PERIOD

        x = .35*m.sin(2*w*t)
        y = .35*m.sin(w*t)
        z = -0.8
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll = 0.0
        pitch = 0.0
        yaw = 0.0

        r = np.array([[x, y, z, vx, vy, vz, roll, pitch, yaw]]).T
        r_final = np.tile(r, (1, self.num_steps))

        return r_final
    
    def fig8_vert_ref_func_short(self): #Returns A Short Figure 8 Reference Trajectory in Vertical Plane ([x,y,z,yaw])
        """ Returns a short figure 8 reference trajectory in vertical plane. """
        print(f"fig8_vert_ref_func_short")

        t = self.time_from_start + self.T_lookahead
        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637
        w = 2*m.pi / PERIOD

        x = 0.0
        y = .35*m.sin(w*t)
        z = -1*(.35*m.sin(2*w*t) + 0.8)
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll = 0.0
        pitch = 0.0
        yaw = 0.0

        r = np.array([[x, y, z, vx, vy, vz, roll, pitch, yaw]]).T
        r_final = np.tile(r, (1, self.num_steps))

        return r_final
    
    def fig8_vert_ref_func_tall(self): #Returns A Tall Figure 8 Reference Trajectory in Vertical Plane ([x,y,z,yaw])
        """ Returns a tall figure 8 reference trajectory in vertical plane. """
        print(f"fig8_vert_ref_func_tall")

        t = self.time_from_start + self.T_lookahead
        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637
        w = 2*m.pi / PERIOD

        x = 0.0
        y = .35*m.sin(2*w*t)
        z = -1*(.35*m.sin(w*t)+0.8)
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll = 0.0
        pitch = 0.0
        yaw = 0.0

        r = np.array([[x, y, z, vx, vy, vz, roll, pitch, yaw]]).T
        r_final = np.tile(r, (1, self.num_steps))

        return r_final

    def helix(self): #Returns Helix Reference Trajectory ([x,y,z,yaw])
        """ Returns helix reference trajectory. """
        print(f"helix")
        
        t = self.time_from_start + self.T_lookahead
        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637
        w = 2*m.pi / PERIOD
        
        PERIOD_Z = 13
        w_z = 2*m.pi / PERIOD_Z
        z0 = 0.7
        height_variance = 0.3

        x = .6*m.cos(w*t)
        y = .6*m.sin(w*t)
        z = -1 * (z0 + height_variance * m.sin(w_z * t))
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll = 0.0
        pitch = 0.0
        yaw = 0.0

        r = np.array([[x, y, z, vx, vy, vz, roll, pitch, yaw]]).T
        r_final = np.tile(r, (1, self.num_steps))

        return r_final
    
    def helix_spin(self):
        """ Returns helix reference trajectory while yawing. """
        print(f"helix_spin")
        
        t = self.time_from_start + self.T_lookahead
        PERIOD = 13
        w = 2*m.pi / PERIOD

        PERIOD_Z = 13
        w_z = 2*m.pi / PERIOD_Z
        z0 = 0.7
        height_variance = 0.3

        SPIN_PERIOD = 15

        x = .6*m.cos(w*t)
        y = .6*m.sin(w*t)
        z = -1 * (z0 + height_variance * m.sin(w_z * t))
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll = 0.0
        pitch = 0.0
        yaw = t / (SPIN_PERIOD / (2*m.pi))

        r = np.array([[x, y, z, vx, vy, vz, roll, pitch, yaw]]).T
        r_final = np.tile(r, (1, self.num_steps))

        return r_final


    def spiral_staircases_old(self, num):
        if not self.sim:
            print("spiral trajectories not YET available for hardware")
            if num > 3:
                print("spiral modes above 3 not available for hardware")
                exit(1)
            exit(1)

        t = self.time_from_start + self.T_lookahead  

        # For x and y elements of spiral staircase
        amplitude_xy = 0.8
        desired_xy_period = 4 #6+ for hardware
        w_xy = (2*m.pi) / desired_xy_period

        # For height element of spiral staircase 
        amplitude_h = 0.8
        buffer = 0.4
        desired_rise_period = desired_xy_period * 2.5 #3+ for hardware with 6+ for xy period
        w_rise = (2*m.pi) / desired_rise_period

        # For Yawing while spiraling
        desired_yaw_period = desired_xy_period * 2
        w_yaw = (2*m.pi) / desired_yaw_period

        circle_period_hardware = 11 # maybe 8?
        w_xy_hardware = 2*m.pi / circle_period_hardware

        vert_period_hardware = circle_period_hardware * 3
        w_vert_hardware = 2*m.pi / vert_period_hardware

        yaw_multiplier_hardware = 1.5 #maybe 2, or 1.5 if you get brave
        w_yaw_hardware = 2*m.pi / (circle_period_hardware*3*yaw_multiplier_hardware)

        spiral_dict = { #4+ are for sim only in case any of this makes it onto hardware
            1: np.array([[0.0, 0.0, -1*( -.8*m.sin((2*m.pi/(6*3))*t) + (0.8+0.4) ),     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]).T,
            2: np.array([[0.8*m.cos(w_xy_hardware*t), 0.8*m.sin(w_xy_hardware*t), -1*( 0.8*m.sin(w_vert_hardware*t) + (0.8+0.4) ),     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]).T,
            3: np.array([[0.8*m.cos(w_xy_hardware*t), 0.8*m.sin(w_xy_hardware*t), -1*( 0.8*m.sin(w_vert_hardware*t) + (0.8+0.4) ),     0.0, 0.0, 0.0,   0.0, 0.0, w_yaw_hardware*t]]).T,
            4: np.array([[0.8*m.cos((2*m.pi/4)*t), 0.8*m.sin((2*m.pi/4)*t), -1*( 0.8*m.sin((2*m.pi/(4*2.5))*t) + (0.8+0.4) ),     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]).T,
            5: np.array([[0.8*m.cos((2*m.pi/4)*t), 0.8*m.sin((2*m.pi/4)*t), -1*( 0.8*m.sin((2*m.pi/(4*2.5))*t) + (0.8+0.4) ),     0.0, 0.0, 0.0,   0.0, 0.0, (2*m.pi/ (4*2.5*2.5))*t]]).T,
            6: np.array([[amplitude_xy*m.cos(w_xy*t), amplitude_xy*m.sin(w_xy*t), -1*( amplitude_h*m.sin(w_rise*t) + (amplitude_h+buffer) ),     0.0, 0.0, 0.0,   0.0, 0.0, w_yaw*t]]).T,
        }


        if num > len(spiral_dict) or num < 1:
            print(f"spiral_dict #{num} not found")
            exit(0)
    
        print(f"spiral_dict #{num}")

        r = spiral_dict.get(num)
        r_final = np.tile(r, (1, self.num_steps))
        return r_final
    
    def yaw_ref_old(self, num):
        if not self.sim:
            print("changing yaw not YET available for hardware")
            if num > 3:
                print("yaw modes above 3 not available for hardware")
                exit(1)
            exit(1)      
        t = self.time_from_start + self.T_lookahead

        # For Yawing
        desired_yaw_period = 6
        w_yaw = (2*m.pi) / desired_yaw_period

        # For circles while yawing
        amplitude_xy = 0.8
        desired_xy_period = 6 #6+ for hardware
        w_xy = (2*m.pi) / desired_xy_period

        hardware_period = 11 # maybe 8?
        w_xy_hardware = 2*m.pi / hardware_period
        yaw_hardware_multiplier = 1.5 #maybe 2, or 1.5 if you get brave
        w_yaw_hardware = 2*m.pi / (hardware_period*3*yaw_hardware_multiplier)
        yaw_dict = {
            1: np.array([[0.0, 0.0, -0.6,     0.0, 0.0, 0.0,   0.0, 0.0, w_yaw_hardware*t]]).T,
            2: np.array([[0.8, 0.8, -0.8,     0.0, 0.0, 0.0,   0.0, 0.0, w_yaw_hardware*t]]).T,
            3: np.array([[0.8*m.cos(w_xy_hardware*t), 0.8*m.sin(w_xy_hardware*t), -0.8,     0.0, 0.0, 0.0,   0.0, 0.0, w_yaw_hardware*t]]).T,
            4: np.array([[amplitude_xy*m.cos(w_xy*t), amplitude_xy*m.sin(w_xy*t), -0.8,     0.0, 0.0, 0.0,   0.0, 0.0, w_yaw*t]]).T,
        }
        if num > len(yaw_dict) or num < 1:
            print(f"yaw_dict #{num} not found")
            exit(1)
    
        r = yaw_dict.get(num)
        r_final = np.tile(r, (1, self.num_steps))
        return r_final



# ~~ Entry point of the code -> Initializes the node and spins it. Also handles exceptions and logging ~~
def main(args=None):
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    logger = None

    def shutdown_logging(*args):
        print("\nInterrupt/Error/Termination Detected, Triggering Logging Process and Shutting Down Node...")
        if logger:
            logger.log(offboard_control)
        offboard_control.destroy_node()
        rclpy.shutdown()
    # Register the signal handler for Ctrl+C (SIGINT)
    # signal.signal(signal.SIGINT, shutdown_logging)

    try:
        print(f"\nInitializing ROS 2 node: '{__name__}' for offboard control")
        logger = Logger([sys.argv[1]])  # Create logger with passed filename
        rclpy.spin(offboard_control)    # Spin the ROS 2 node
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected (Ctrl+C), exiting...")
    except Exception as e:
        print(f"\nError in main: {e}")
        traceback.print_exc()
    finally:
        shutdown_logging()
        print("\nNode has shut down.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError in __main__: {e}")
        traceback.print_exc()