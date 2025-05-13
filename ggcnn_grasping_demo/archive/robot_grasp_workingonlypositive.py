import time
import math
import threading
import numpy as np
from .helpers.matrix_funcs import euler2mat, convert_pose
from xarm.wrapper import XArmAPI


# 3次滑动平均，每组输入为4个变量。[x,y,z,yaw]
class Averager():
    def __init__(self, inputs, time_steps):
        self.buffer = np.zeros((time_steps, inputs))
        self.steps = time_steps
        self.curr = 0
        self.been_reset = True

    def update(self, v):
        if self.steps == 1:
            self.buffer = v
            return v
        self.buffer[self.curr, :] = v
        self.curr += 1
        if self.been_reset:
            self.been_reset = False
            while self.curr != 0:
                self.update(v)
        if self.curr >= self.steps:
            self.curr = 0
        pos = self.buffer.mean(axis=0)
        return pos

    def evaluate(self):
        if self.steps == 1:
            return self.buffer
        return self.buffer.mean(axis=0)

    def reset(self):
        self.buffer *= 0
        self.curr = 0
        self.been_reset = True


class MinPos():
    def __init__(self, inputs, time_steps):
        self.buffer = np.zeros((time_steps, inputs))
        self.steps = time_steps
        self.curr = 0
        self.been_reset = True
        self.inputs = inputs
        self.prev_pos = [0, 0, 0, 0]

    def update(self, v):
        if self.steps == 1:
            self.buffer = v
            return v
        self.buffer[self.curr, :] = v
        self.curr += 1
        if self.been_reset:
            self.been_reset = False
            while self.curr != 0:
                self.update(v)
        if self.curr >= self.steps:
            self.curr = 0
        min_inx = 0
        min_dis = 9999
        for i in range(self.steps):
            dis = pow(self.buffer[i][0] - self.prev_pos[0], 2) + pow(self.buffer[i][1] - self.prev_pos[1], 2) + pow(self.buffer[i][2] - self.prev_pos[2], 2)
            if dis < min_dis:
                min_dis = dis
                min_inx = i
        self.prev_pos[0] = self.buffer[min_inx][0]
        self.prev_pos[1] = self.buffer[min_inx][1]
        self.prev_pos[2] = self.buffer[min_inx][2]
        self.prev_pos[3] = self.buffer[min_inx][3]
        return self.prev_pos
    
    def evaluate(self):
        if self.steps == 1:
            return self.buffer
        return self.prev_pos

    def reset(self):
        self.buffer *= 0
        self.curr = 0
        self.prev = 0
        self.been_reset = True


class RobotGrasp(object):    
    CURR_POS = [300, 0, 350, 180, 0, 0]
    GOAL_POS = [0, 0, 0, 0, 0, 0]

    SERVO = True
    GRASP_STATUS = 0

    def __init__(self, robot_ip, ggcnn_cmd_que, euler_eef_to_color_opt, euler_color_to_depth_opt, grasping_range, detect_xyz, gripper_z_mm, release_xyz, grasping_min_z, use_init_pos = False, stop_arm_on_success = False, on_trigger_mode = False, hori_pickup=False):
        self.arm = XArmAPI(robot_ip)
        self.ggcnn_cmd_que = ggcnn_cmd_que
        self.euler_eef_to_color_opt = euler_eef_to_color_opt
        self.euler_color_to_depth_opt = euler_color_to_depth_opt
        self.grasping_range = grasping_range
        self.use_init_pos = use_init_pos
        self.stop_arm_on_success = stop_arm_on_success # stop the arm upon successful grasp
        self.on_trigger_mode = on_trigger_mode # stow the arm if this is set to true
        if self.use_init_pos:
            self.detect_xyz = None
        else:
            self.detect_xyz = detect_xyz
        self.detect_rpy = None
        self.init_j_pose = None
        self.init_pose = None
        self.gripper_z_mm = gripper_z_mm
        self.release_xyz = release_xyz
        self.grasping_min_z = grasping_min_z
        # self.pose_averager = Averager(4, 3)
        self.pose_averager = MinPos(4, 3)
        self.is_ready = False
        self.alive = True
        self.last_grasp_time = 0
        self.pos_t = threading.Thread(target=self.update_pos_loop, daemon=True)
        self.pos_t.start()
        self.ggcnn_t = threading.Thread(target=self.handle_ggcnn_loop, daemon=True)
        self.ggcnn_t.start()
        self.check_t = threading.Thread(target=self.check_loop, daemon=True)
        self.check_t.start()
        
        self.arm_resetting = False # tag to indicate when in process to reset arm
        self.xarm_setup = True # tag to indicate if it is setup or reset for arm
        self.hori_pickup = hori_pickup # tag to indicate top-down pickup or horizontal pickup
        self.dist_valid = True # tag to stop looking for additional distance values when approaching 
    
    def is_alive(self):
        return self.alive
    
    def handle_ggcnn_loop(self):
        while self.arm.connected and self.alive:
            cmd = self.ggcnn_cmd_que.get()
            self.grasp(cmd)
            # print("cmd",cmd)
            time.sleep(0.15)

    def stop_motion(self):
        if not self.SERVO:
            return
        self.SERVO = False
        self.GRASP_STATUS = 0
        print('>>>>', self.CURR_POS, self.GOAL_POS)
        self.arm.set_state(4)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        # arm.set_position(*self.GOAL_POS, wait=True)
        time.sleep(0.1)
        self.arm.set_gripper_position(-10, wait=True)
        time.sleep(0.5)
        # check if the grasp is empty or not
        self.arm.set_servo_angle(angle=self.init_j_pose, speed=50, mvacc=1000, wait=True) # going to grasp position but in joint space to avoid IK errors
        if not(self.arm.get_gripper_position()[1]) < 0: # only go to recepticle if gripped something
            if self.stop_arm_on_success: # stop grasp code if enabled
                # go to stowed
                self.arm.set_mode(0)
                self.arm.set_state(0)
                self.arm.set_servo_angle(angle=[0,90,0,0,0,0], speed=50, wait=True)
                self.alive = False
                return 
            self.arm.set_position(x=self.release_xyz[0], y=self.release_xyz[1], roll=180, pitch=0, yaw=0, speed=200, wait=True)
            self.arm.set_position(z=self.release_xyz[2], speed=100, wait=True)
        # time.sleep(3)
        # input('Press Enter to Complete')

        # Open Fingers
        self.arm.set_gripper_position(850, wait=True)
        # time.sleep(5)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_servo_angle(angle=self.init_j_pose, speed=50, mvacc=1000, wait=True) # going to grasp position but in joint space to avoid IK errors
        self.dist_valid = True # reset this after grasping attempt
        self.pose_averager.reset()
        self.arm.set_mode(7)
        self.arm.set_state(0)
        time.sleep(2)

        # input('Press Enter to Start')
        self.SERVO = True
        self.last_grasp_time = time.monotonic()
    
    def xarm_init(self):
        self.arm.clean_error()
        self.arm.clean_warn()
        time.sleep(1)
        self.arm.set_mode(0)
        time.sleep(1)
        self.arm.motion_enable(True)
        self.arm.set_state(0)
        time.sleep(0.5)
        # print("self.arm.error_code: ", self.arm.error_code)
        if self.init_j_pose is None:
            # go to initial position
            _, init_pos = tuple(self.arm.get_initial_point())
            self.init_j_pose = init_pos
        if self.xarm_setup:   
            self.arm.set_servo_angle(angle=self.init_j_pose,wait=True,is_radian=False)
            self.xarm_setup = False
            self.arm.set_gripper_enable(True)
            self.arm.set_gripper_position(850)
        else:
            if self.arm.error_code == 0:
                print("Resetting to init position")
                self.arm.set_servo_angle(angle=self.init_j_pose,wait=True,is_radian=False)
                self.arm.set_gripper_enable(True)
                self.arm.set_gripper_position(850)
            else:
                print("Skipped resetting")

    def update_pos_loop(self):
        # print("*** STARTED update_pos_loop ***")
        # print("Previous error/warn code: ", self.arm.get_err_warn_code())
        
        # Initialize robot if needed
        if self.init_pose is None:
            self.xarm_init()
            self.arm.set_servo_angle(angle=self.init_j_pose, wait=True, is_radian=False)
            time.sleep(0.5)
            _, init_pose = self.arm.get_position(is_radian=True)
            self.init_pose = np.array(init_pose, dtype=np.float32)
            self.detect_xyz = ([init_pose[0], init_pose[1], init_pose[2]])
            self.detect_rpy = ([init_pose[3], init_pose[4], init_pose[5]])

            if self.on_trigger_mode:
                # go back to stow position after initial setup/check arm
                self.arm.set_servo_angle(angle=[0, 90, 0, 0, 0, 0], wait=True, is_radian=False)        
                time.sleep(0.5)

        self.SERVO = True
        self.arm.set_mode(7)
        self.arm.set_state(0)
        time.sleep(0.5)

        self.is_ready = True
        self.arm_resetting = False
        
        # Main continuous loop that handles all states
        while self.arm.connected and self.alive:
            # print("Current error code:", self.arm.error_code)
            
            if self.arm.error_code == 0:
                # Normal operation - monitor position
                try:
                    _, pos = self.arm.get_position()
                    # warn_code = self.arm.get_err_warn_code()
                    # print("Error/warn code: ", warn_code)
                    self.CURR_POS = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]
                except Exception as e:
                    print("Exception getting position:", str(e))
                    
            elif self.arm.error_code == 31:
                # Collision detected
                self.arm_resetting = True
                print('Collision detected, resetting arm')
                
                # Our reset procedure
                # print("Before xarm_init error code:", self.arm.error_code)
                self.xarm_init()
                # print("After xarm_init error code:", self.arm.error_code)
                
                # If we've cleared the error, reset the servo mode
                if self.arm.error_code == 0:
                    # print("Successfully reset arm, returning to servo mode")
                    self.arm.set_mode(7)
                    self.arm.set_state(0)
                    time.sleep(0.5)
                
                self.arm_resetting = False
            
            elif self.arm.error_code == 23:
                # joint angle limit exceeded 
                self.arm_resetting = True
                print('Joint angle limit exceeded, resetting arm')
                
                # Our reset procedure
                # print("Before xarm_init error code:", self.arm.error_code)
                self.xarm_init()
                # print("After xarm_init error code:", self.arm.error_code)
                
                # If we've cleared the error, reset the servo mode
                if self.arm.error_code == 0:
                    # print("Successfully reset arm, returning to servo mode")
                    self.arm.set_mode(7)
                    self.arm.set_state(0)
                    time.sleep(0.5)
                
                self.arm_resetting = False
            
            elif self.arm.error_code == 21 or self.arm.error_code == 24:
                # kinematic error
                self.arm_resetting = True
                print('Kinematic error, resetting arm')
                
                # Our reset procedure
                # print("Before xarm_init error code:", self.arm.error_code)
                self.xarm_init()
                # print("After xarm_init error code:", self.arm.error_code)
                
                # If we've cleared the error, reset the servo mode
                if self.arm.error_code == 0:
                    # print("Successfully reset arm, returning to servo mode")
                    self.arm.set_mode(7)
                    self.arm.set_state(0)
                    time.sleep(0.5)
                
                self.arm_resetting = False

            else:
                # Other errors
                print(f"Handling error code: {self.arm.error_code}")
                self.arm_resetting = True
                
                # Try to clear the error
                self.arm.clean_error()
                self.arm.clean_warn()
                time.sleep(1)
                
                # If still error, try further recovery
                if self.arm.error_code != 0:
                    print(f"Error {self.arm.error_code} persists, trying full reset")
                    self.xarm_init()
                
                # If we've cleared the error, reset the servo mode
                if self.arm.error_code == 0:
                    print("Successfully recovered, returning to servo mode")
                    self.arm.set_mode(7)
                    self.arm.set_state(0)
                    time.sleep(0.5)
                else:
                    print(f"Could not recover from error {self.arm.error_code}")
                    if self.arm.error_code == 1:  # Errors not cleared
                        print("Special handling for error code 1: force reset")
                        # Force a more aggressive reset for error code 1
                        self.arm.clean_error()
                        self.arm.clean_warn()
                        time.sleep(1)
                        self.arm.motion_enable(True)
                        self.arm.set_mode(0)
                        self.arm.set_state(0)
                        time.sleep(1)
                        self.arm.set_mode(7)
                        self.arm.set_state(0)
                    else:
                        # Unrecoverable error
                        print('Unrecoverable error - exiting')
                        self.alive = False
                        
                self.arm_resetting = False
                
            time.sleep(0.1)  # Small sleep to prevent CPU hogging
            
        # print("update_pos_loop exited - disconnecting")
        self.arm.disconnect()
    
    def check_loop(self):
        while self.arm.connected and self.arm.error_code == 0:
            # print("check thread running")
            self._check()
            time.sleep(0.01)

    def _check(self):
        if not self.is_ready or not self.alive:
            return
        x = self.CURR_POS[0]
        y = self.CURR_POS[1]
        z = self.CURR_POS[2]
        roll = self.CURR_POS[3]
        pitch = self.CURR_POS[4]
        yaw = self.CURR_POS[5]
        # reset to start position if moved outside of boundary
        if (time.monotonic() - self.last_grasp_time) > 5 or x < self.grasping_range[0] or x > self.grasping_range[1] or y < self.grasping_range[2] or y > self.grasping_range[3]:
            if (time.monotonic() - self.last_grasp_time) > 5 \
                and abs(x-self.detect_xyz[0]) < 2 and abs(y-self.detect_xyz[1]) < 2 and abs(z-self.detect_xyz[2]) < 2 \
                and abs(abs(roll)-180) < 2 and abs(pitch) < 2 and abs(yaw) < 2:
                self.last_grasp_time = time.monotonic()
                return
            print("restart triggered current state: ", x,y,z,roll,pitch,yaw)
            self.SERVO = False
            self.arm.set_state(4)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(1)
            # going to grasp position but in joint space to avoid IK errors
            self.arm.set_servo_angle(angle=self.init_j_pose, speed=50, mvacc=1000, wait=True) 
            time.sleep(0.25)
            self.pose_averager.reset()
            self.arm.set_mode(7)
            self.arm.set_state(0)
            time.sleep(0.5)
            self.GRASP_STATUS = 0
            self.SERVO = True
            self.last_grasp_time = time.monotonic()
            return
        
        x_diff_abs = abs(self.CURR_POS[0] - self.GOAL_POS[0])
        y_diff_abs = abs(self.CURR_POS[1] - self.GOAL_POS[1])
        z_diff_abs = abs(self.CURR_POS[2] - self.GOAL_POS[2])
        if not self.hori_pickup: # use gripper_z_mm based on orientation of pickup
            z_val_min = self.grasping_min_z - self.gripper_z_mm
        # print("x_diff_abs: ", x_diff_abs)
        # print("y_diff_abs: ", y_diff_abs)
        # print("z_diff_abs: ", z_diff_abs)
        if x_diff_abs < 5 and y_diff_abs < 5:
                self.GRASP_STATUS = 1
        if not self.hori_pickup: # only relevant for veritical pickup to avoid collision with surface
            # stop if current z is less than threshold
            if  (z_diff_abs < 1) or (z < z_val_min): # grasping and z being out of bounds
                self.stop_motion()
        else:
            if (x_diff_abs < 1) or (y_diff_abs < 1) or (z < self.grasping_min_z): # case of grasping and z being out of bounds
                self.stop_motion()
        
    def get_eef_pose_m(self):
        _, eef_pos_mm = self.arm.get_position(is_radian=True)
        eef_pos_m = [eef_pos_mm[0]*0.001, eef_pos_mm[1]*0.001, eef_pos_mm[2]*0.001, eef_pos_mm[3], eef_pos_mm[4], eef_pos_mm[5]]
        return eef_pos_m

    def grasp(self, data):
        if not self.alive or not self.is_ready or not self.SERVO:
            return

        d = list(data)
        # print("length of data sent: ", len(d))
        if len(d) <= 6: # 7th element is the eef pose
            # PBVS Method.
            # print("getting pose after grasp img is calculated")
            euler_base_to_eef = self.get_eef_pose_m()
            # print("grasp data eef: ", euler_base_to_eef)
        else:
            # print("grasp data eef: ", d)
            euler_base_to_eef = d[6]
            print("grasp data eef 6: ", euler_base_to_eef)
        # print(d) # ang in radians
        # print([d[0], d[1], d[2], 0, 0, -d[3]])
        # if d[2] > 0.35:  # Min effective range of the oakdpro.
        if d[2] > 0.2 and self.dist_valid:  # Min effective range of the realsense.
            if not self.hori_pickup:
                gp = [d[0], d[1], d[2], 0, 0, -d[3]] # xyz00(angle of grasp) in meter
            else:
                gp = [d[0], d[1], d[2], d[3], 0, 0] # xyz00(angle of grasp) in meter
            # print("gp", gp)
            # Calculate Pose of Grasp in Robot Base Link Frame
            # Average over a few predicted poses to help combat noise.
            # print("gp", gp)
            mat_depthOpt_in_base = euler2mat(euler_base_to_eef) * euler2mat(self.euler_eef_to_color_opt) * euler2mat(self.euler_color_to_depth_opt)
            # print(euler2mat(self.euler_eef_to_color_opt))
            # print(mat_depthOpt_in_base)
            # print("euler frame", rot_to_rpy(mat_depthOpt_in_base[0:3,0:3]))
            gp_base = convert_pose(gp, mat_depthOpt_in_base)
            
            if not self.hori_pickup:
                if gp_base[5] < -np.pi:
                    gp_base[5] += np.pi
                elif gp_base[5] > np.pi:
                    gp_base[5] -= np.pi
                av = self.pose_averager.update(np.array([gp_base[0], gp_base[1], gp_base[2], gp_base[5]]))
            else:
                print("Original angle: ", math.degrees(gp_base[3]))
                # Normalize using modulo to get equivalent angle in -180° to +180° range
                gp_base[3] = ((gp_base[5] + np.pi) % (2 * np.pi)) - np.pi
                print("After first normalization: ", math.degrees(gp_base[3]))
                # if gp_base[3] < -np.pi/2:
                #     gp_base[3] = -np.pi/2
                # elif gp_base[3] > np.pi/2:
                #     gp_base[3] = np.pi/2
                # print("Final normalized angle: ", math.degrees(gp_base[3]))
                av = self.pose_averager.update(np.array([gp_base[0], gp_base[1], gp_base[2], gp_base[3]]))
        else:
            self.dist_valid = False # stop reading from the camera
            # gp_base = geometry_msgs.msg.Pose()
            gp_base = [0]*6
            av = self.pose_averager.evaluate()
        # print("av: ", av)
        # Average pose in base frame.
        # print("Current Yaw: ", self.CURR_POS[5])
        # print("ang init: ", math.degrees(av[3]))
        
        # if not self.hori_pickup:
        #     ang = av[3] - np.pi/2  # We don't want to align, we want to grip.
        # else:
        #     ang = av[3]  # We don't want to align, we want to grip.
        # print("ang after pi/2: ", math.degrees(ang))
        # # Normalize the angle to avoid full rotations
        # while ang < -np.pi:
        #     ang += 2*np.pi
        # while ang > np.pi:
        #     ang -= 2*np.pi
        # # print("normalized ang: ", math.degrees(ang))
        # # calculate the point with the least differece

        # if not self.hori_pickup:
        #     ang = ang + np.pi
        #     while ang < -np.pi:
        #         ang += 2*np.pi
        #     while ang > np.pi:
        #         ang -= 2*np.pi
        #     print("ang after +pi: ", math.degrees(ang))
        # if not self.hori_pickup: # add offset to the x or z dimension for improved grip
        #     GOAL_POS = [av[0] * 1000, av[1] * 1000, av[2] * 1000 + self.gripper_z_mm, self.CURR_POS[3], self.CURR_POS[4], math.degrees(ang)]
        #     if abs(GOAL_POS[2]) < self.gripper_z_mm + 10:
        #         # print('[IG]', GOAL_POS)
        #         return
        # else:
        #     GOAL_POS = [av[0] * 1000 + self.gripper_z_mm, av[1] * 1000, av[2] * 1000,  math.degrees(ang), self.CURR_POS[4], self.CURR_POS[5]]
        # print("GOAL_POS", GOAL_POS)
        # print(GOAL_POS[2])
        # GOAL_POS[2] = min(abs(euler_base_to_eef[2]-d[2]), euler_base_to_eef[2]) # forced to only pick up from table
        # GOAL_POS[2] = GOAL_POS[2]*1000
        # print(GOAL_POS)
        # Get current angle of the robot (in degrees)

        # Process initial angle based on pickup type
        ang = av[3] - np.pi/2  # For non-horizontal pickup, adjust by -90°
        
        # Get current angle of the robot (in radians)
        current_angle_rad = math.radians(self.CURR_POS[5]) if not self.hori_pickup else math.radians(self.CURR_POS[3])
        # print("Current angle (radians): ", current_angle_rad)
        print("Current angle (degrees): ", self.CURR_POS[5] if not self.hori_pickup else self.CURR_POS[3])

        # print("Initial angle (radians): ", ang)
        print("Initial angle (degrees): ", math.degrees(ang))

        # # Normalize the angle to avoid full rotations
        while ang < -np.pi:
            ang += 2*np.pi
        while ang > np.pi:
            ang -= 2*np.pi

        # print("Normalized angle (radians): ", ang)
        print("Normalized angle (degrees): ", math.degrees(ang))

        # Create another angle option (original angle + 180°)
        ang_alt = ang + np.pi
        # Normalize alternative angle
        while ang_alt < -np.pi:
            ang_alt += 2*np.pi
        while ang_alt > np.pi:
            ang_alt -= 2*np.pi

        # Calculate rotation distances for both angles
        dist_orig = abs(ang - current_angle_rad)
        dist_alt = abs(ang_alt - current_angle_rad)

        # Handle wrap-around cases
        if dist_orig > np.pi:
            dist_orig = 2*np.pi - dist_orig
        if dist_alt > np.pi:
            dist_alt = 2*np.pi - dist_alt

        # print("Distance to original angle: ", dist_orig)
        # print("Distance to +180° angle: ", dist_alt)

        # Choose angle with smallest rotation
        if dist_orig <= dist_alt:
            final_ang = ang
        else:
            final_ang = ang_alt

        # print("Final angle (radians): ", final_ang)
        print("Final angle (degrees): ", math.degrees(final_ang))

        # Set goal position based on pickup orientation
        if not self.hori_pickup:  # add offset to z dimension for vertical pickup
            GOAL_POS = [av[0] * 1000, av[1] * 1000, av[2] * 1000 + self.gripper_z_mm, 
                    self.CURR_POS[3], self.CURR_POS[4], math.degrees(final_ang)]
            if abs(GOAL_POS[2]) < self.gripper_z_mm + 10:
                # print('[IG]', GOAL_POS)
                return
        else:  # add offset to x dimension for horizontal pickup
            GOAL_POS = [av[0] * 1000 + self.gripper_z_mm, av[1] * 1000, av[2] * 1000, 
                    math.degrees(final_ang), self.CURR_POS[4], self.CURR_POS[5]]

        GOAL_POS[0] = np.clip(self.grasping_range[0], self.grasping_range[1], GOAL_POS[0])
        GOAL_POS[1] = np.clip(self.grasping_range[2], self.grasping_range[3], GOAL_POS[1])
        GOAL_POS[2] = max(GOAL_POS[2],self.grasping_min_z)
        # print(GOAL_POS)
        self.last_grasp_time = time.monotonic()

        # print("GOAL_POS", GOAL_POS)
        # print("CUR_POS", self.CURR_POS)
        if not self.arm_resetting and self.arm.error_code == 0: # only go to pose if arm is not resetting
            # self.GOAL_POS = GOAL_POS
            # self.arm.set_position(*self.GOAL_POS, speed=50, mvacc=500, wait=False)
            if self.GRASP_STATUS == 0:
                self.GOAL_POS = GOAL_POS
                print("GOAL_POS", GOAL_POS)
                print("CUR_POS", self.CURR_POS)
                # _,GOAL_POS_J = self.arm.get_inverse_kinematics(self.GOAL_POS, return_is_radian=False)
                # GOAL_POS_J = np.round(np.array(GOAL_POS_J[:-1],dtype=np.float32),2)
                # print("GOAL_POS_J", GOAL_POS_J)
                # self.arm.set_mode(0)
                # self.arm.set_state(0)
                # self.arm.set_servo_angle(angle=GOAL_POS_J, speed=25, mvacc=1000, wait=False) 
                # align the robot before approaching in z
                if GOAL_POS[0] != self.release_xyz[0]:
                    xy_bin = abs(self.GOAL_POS[0] - self.CURR_POS[0]) > 25 and abs(self.GOAL_POS[1] - self.CURR_POS[1]) > 25 # mm
                    # print("xy_bin", xy_bin)
                    # ang_bin = (abs(self.GOAL_POS[3] - self.CURR_POS[3]) > 5) and (abs(self.GOAL_POS[4] - self.CURR_POS[4])  > 5) and (abs(self.GOAL_POS[5] - self.CURR_POS[5]) > 5) # deg
                    # print("ang_bin", ang_bin)
                    # print(self.CURR_POS)
                    # print(self.GOAL_POS)
                    # print((-self.CURR_POS[2]+self.GOAL_POS[2])*0.5+self.CURR_POS[2])
                    if not self.hori_pickup:
                            self.arm.set_position(x=self.GOAL_POS[0], y=self.GOAL_POS[1], z = (-self.CURR_POS[2]+self.GOAL_POS[2])*0.5+self.CURR_POS[2],
                                                roll=self.GOAL_POS[3], pitch=self.GOAL_POS[4], yaw=self.GOAL_POS[5], 
                                                speed=75, mvacc=750, wait=False)
                    else: # adjust distance first and angle after
                        if xy_bin:
                            # self.arm.set_position(z=self.GOAL_POS[2], y = (-self.CURR_POS[1]+self.GOAL_POS[1])*0.75+self.CURR_POS[1], 
                            #                     x = (-self.CURR_POS[0]+self.GOAL_POS[0])*0.75+self.CURR_POS[0],
                            #                     roll=self.CURR_POS[3], pitch=self.CURR_POS[4], yaw=self.CURR_POS[5], 
                            #                     speed=75, mvacc=750, wait=False)
                            self.arm.set_position(z=self.GOAL_POS[2], y = self.GOAL_POS[1], 
                                                x = self.GOAL_POS[0],
                                                roll=self.CURR_POS[3], pitch=self.CURR_POS[4], yaw=self.CURR_POS[5], 
                                                speed=75, mvacc=750, wait=False)
                        else:
                            self.arm.set_position(x=self.GOAL_POS[0], y=self.GOAL_POS[1], z = self.GOAL_POS[2],
                                                roll=self.GOAL_POS[3], pitch=self.GOAL_POS[4], yaw=self.GOAL_POS[5], 
                                                speed=75, mvacc=750, wait=False)

                else:
                    self.arm.set_position(x=self.GOAL_POS[0], y=self.GOAL_POS[1], z = self.GOAL_POS[2],
                                            roll=self.GOAL_POS[3], pitch=self.GOAL_POS[4], yaw=self.GOAL_POS[5], 
                                            speed=50, mvacc=1000, wait=False)
                # self.pose_averager.reset()
                # self.pose_averager.update(av)

            elif self.GRASP_STATUS == 1:
                self.GOAL_POS = GOAL_POS
                self.arm.set_position(*self.GOAL_POS, speed=50, acc=1000, wait=False)
