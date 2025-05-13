import time
import signal
from xarm.wrapper import XArmAPI
import numpy as np
import sys

ARM_IP ='172.16.0.13'
pos_j_arr = np.array([
                    # [-13,9,-16,0,-25,-13], # big box start
                    [-17.3,21.4,-7,0,-28.5,-17], # small box start
                    [-22,-17,-76,0,-60,-24],
                    #  [-22,-17,-76,0,-59,70],
                    [-22,-20,-70,0,-50,70],
                    [-19,-34,-91,0,-58,74],
                    [-25,-23,-52,0,-29,67],
                    [-25,-19,-53,0,-34,67],
                    [-32,-6,-37,0,-31,60],
                    [-68, -34, -100,0,-69,-160],
                    [-68,-41,-88,0,-47,-160]
                    ])
pos_j_arr_test = np.array([
                    [-68, -34, -100,0,-69,-160],
                    [-68,-41,-88,0,-47,-160]])
init_pos = np.array([0,0,0,0,0,0])
class ARMTEST():
    def __init__(self):
        """Initialize the drive train before each test"""
        try:
            self.arm = self.xarm_init()
            self.arm_enable=True
        except Exception as e:
            self.fail(f"Failed to initialize drive train: {e}")

    def xarm_init(self, ip = ARM_IP):
        global init_pos
        """Initialize the drive train before each test"""
        arm = None
        try:
            arm = XArmAPI(ip)
            arm.clean_error()
            arm.clean_warn()
            arm.clean_gripper_error()
            arm.set_collision_tool_model(2) # xarm gripper
            arm.set_gripper_enable(enable=True)
            arm.set_gripper_mode(0)
            arm.motion_enable(enable=True)
            arm.set_mode(0)
            arm.set_state(0)
            arm.set_vacuum_gripper(False, True)
            time.sleep(0.5)
            _, init_pos = tuple(arm.get_initial_point())
            arm.set_servo_angle(angle=init_pos, wait=True, is_radian=False)
            arm.set_vacuum_gripper(False, True)
            print("CAN and drivetrain initialized")
            arm.set_mode(5)
            arm.set_state(0)
            time.sleep(1)
            print("Arm initialized")
        except Exception as e:
            print(f"Failed to initialize arm : {e}")
        return arm

    def sigterm_handler(self, sig, frame):
        print("sigterm recv\n")
        # self.drive.disable()
        sys.exit(0)

    def sigint_handler(self, sig, frame):
        print("sigint recv\n")
        # self.drive.swerveDrive(0,0,0)
        # self.drive.disable()
        sys.exit(0)
    
    def small_box(self):
        # get arm to the intial position for grasping
        self.arm.set_servo_angle(angle=init_pos,wait=True,is_radian=False)
        self.arm.set_servo_angle(angle=pos_j_arr[0],wait=True,is_radian=False)
        self.arm.set_vacuum_gripper(True, True)
        time.sleep(1)
        self.arm.set_servo_angle(angle=pos_j_arr[1],wait=True,is_radian=False)
        self.arm.set_servo_angle(angle=pos_j_arr[2],wait=True,is_radian=False)
        self.arm.set_servo_angle(angle=pos_j_arr[3],wait=False,is_radian=False)
        self.arm.set_servo_angle(angle=pos_j_arr[4],wait=True,is_radian=False)
        self.arm.set_vacuum_gripper(False, True)
        time.sleep(0.5)
        # buffer between second drop off
        self.arm.set_servo_angle(angle=pos_j_arr[3],wait=False,is_radian=False)
        # xarm.arm.set_servo_angle(angle=init_pos,wait=True,is_radian=False)
        self.arm.set_servo_angle(angle=pos_j_arr[4],wait=True,is_radian=False)
        self.arm.set_vacuum_gripper(True, True, delay_sec=1)
        self.arm.set_servo_angle(angle=pos_j_arr[7],wait=True,is_radian=False)
        self.arm.set_servo_angle(angle=pos_j_arr[8],wait=True,is_radian=False)
        self.arm.set_vacuum_gripper(False, True, delay_sec=0.5)
        self.arm.set_servo_angle(angle=pos_j_arr[7],wait=True,is_radian=False)
        self.arm.set_servo_angle(angle=pos_j_arr[1],wait=True,is_radian=False)
        self.arm.set_servo_angle(angle=pos_j_arr[7],wait=True,is_radian=False)
        self.arm.set_servo_angle(angle=pos_j_arr[8],wait=True,is_radian=False)
        self.arm.set_vacuum_gripper(True, True, delay_sec=1)
        #going home
        self.arm.set_servo_angle(angle=pos_j_arr[1],wait=True,is_radian=False)
        self.arm.set_servo_angle(angle=pos_j_arr[0],wait=True,is_radian=False)
        self.arm.set_vacuum_gripper(False, True, delay_sec=0.5)
        
if __name__ == '__main__':
    print("=== Swerve Drive Test Suite ===")
    xarm=ARMTEST()
    # init sig handlers
    signal.signal(signal.SIGINT, xarm.sigint_handler)
    signal.signal(signal.SIGTERM, xarm.sigterm_handler)
    time.sleep(2)

    _, cur_pos = xarm.arm.get_servo_angle(is_radian=False)
    goal_pos = cur_pos
    print("cur_pos: ", cur_pos)
    goal_pos[5] = cur_pos[5] + 10
    print("goal_pos: ", goal_pos)
    xarm.arm.set_mode(0)
    xarm.arm.set_state(0)
    xarm.arm.set_servo_angle(angle=[8.9,31,-6.7,12.3,52.4,10], is_radian=False, speed=500, mvacc=1000, wait=True) 
    exit()
    # arm to the intial position for grasping
    xarm.arm.set_mode(0)
    xarm.arm.set_state(0)
    pos = [6.5,14,-15,8,60,0]
    pos2 = [6.5,14,-15,8,60,-10]
    while True:
        xarm.arm.set_servo_angle(angle=pos,wait=True,is_radian=False)
        time.sleep(1)
        xarm.arm.set_servo_angle(angle=pos2,wait=True,is_radian=False)
        