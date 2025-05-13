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

    def __init__(self, robot_ip, ggcnn_cmd_que, euler_eef_to_color_opt, euler_color_to_depth_opt, grasping_range, detect_xyz, gripper_z_mm, release_xyz, grasping_min_z, use_init_pos = False, stop_arm_on_success = False, on_trigger_mode = False, hori_pickup=False, cutoff_dist=0.3, max_allowable_dist = 2): # dist in m
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
        self.pose_averager = MinPos(4, 2)
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

        # arm control tags - specifically used in horizontal gripping
        # by default it is in 0 mode
        self.xy_ctrl = True
        self.roll_ctrl = False
        self.ggcnn_cutoff_dist = cutoff_dist
        self.max_arm_ctrl_dist = max_allowable_dist

    def is_alive(self):
        return self.alive
    
    def handle_ggcnn_loop(self):
        while self.arm.connected and self.alive:
            cmd = self.ggcnn_cmd_que.get()
            # print("cmd: ",cmd)
            self.grasp(cmd)

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
        print("going to home after gripping from stop_motion")
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
        # lift z to large value to avoid any collisions
        if self.hori_pickup:
            self.arm.set_position(z=600, speed=100, wait=True)
        # time.sleep(5)
        self.arm.set_mode(0)
        print("resetting from stop_motion")
        self.arm.set_servo_angle(angle=self.init_j_pose, speed=50, mvacc=1000, wait=True) # going to grasp position but in joint space to avoid IK errors
        self.dist_valid = True # reset this after grasping attempt
        self.pose_averager.reset()
        self.arm.set_mode(7)
        self.xy_ctrl = True
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
            print("Setting up the arm")
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
            print("Initial Pose Recording")
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
        self.xy_ctrl = True
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
                    pos = [round(val,3) for val in pos]
                    self.CURR_POS = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]
                    # print("In err code, self.CURR_POS: ", self.CURR_POS)
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
                    self.xy_ctrl = True
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
                    self.xy_ctrl = True
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
                    self.xy_ctrl = True
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
                    self.xy_ctrl = True
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
                        time.sleep(1)
                        self.arm.set_mode(7)
                        self.xy_ctrl = True
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
        # reset to start position if moved outside of boundary or grasping timed out and needs to be reset
        # Check if outside boundary or 5-second timeout
        timeout_occurred = (time.monotonic() - self.last_grasp_time) > 5
        outside_boundary = (x < self.grasping_range[0] or x > self.grasping_range[1] or 
                        y < self.grasping_range[2] or y > self.grasping_range[3])
        
        # Check if at detection position
        if not self.hori_pickup:
            # For vertical pickup - check position and orientation
            at_detect_pos = (abs(x-self.detect_xyz[0]) < 2 and 
                            abs(y-self.detect_xyz[1]) < 2 and 
                            abs(z-self.detect_xyz[2]) < 2 and
                            abs(abs(roll)-180) < 2 and 
                            abs(pitch) < 2 and 
                            abs(yaw) < 2)
        else:
            # For horizontal pickup - check position only, ignore orientation
            at_detect_pos = (abs(x-self.detect_xyz[0]) < 2 and 
                            abs(y-self.detect_xyz[1]) < 2 and 
                            abs(z-self.detect_xyz[2]) < 2)
        
        # reset to start position if moved outside of boundary or grasping timed out
        if outside_boundary or timeout_occurred:
            print("outside_boundary: ", outside_boundary)
            print("timeout_occurred: ", timeout_occurred)
            print("at_detect_pos: ", at_detect_pos)
        
            print("restart triggered current state: ", x,y,z,roll,pitch,yaw)
            self.SERVO = False
            self.arm.set_state(4)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(1)
            # going to grasp position but in joint space to avoid IK errors
            print("going home beacuse of boundary/timeout")
            self.arm.set_servo_angle(angle=self.init_j_pose, speed=50, mvacc=1000, wait=True)
            time.sleep(0.25)
            self.pose_averager.reset()
            self.arm.set_mode(7)
            self.xy_ctrl = True
            self.arm.set_state(0)
            time.sleep(0.5)
            self.GRASP_STATUS = 0
            self.SERVO = True
            self.last_grasp_time = time.monotonic()
            self.dist_valid = True
            return
            
        # Update the timer if timeout occurred but we didn't reset
        if timeout_occurred:
            self.last_grasp_time = time.monotonic()
            
        x_diff_abs = abs(self.CURR_POS[0] - self.GOAL_POS[0])
        y_diff_abs = abs(self.CURR_POS[1] - self.GOAL_POS[1])
        z_diff_abs = abs(self.CURR_POS[2] - self.GOAL_POS[2])
        if not self.hori_pickup: # use gripper_z_mm based on orientation of pickup
            z_val_min = self.grasping_min_z - self.gripper_z_mm
        # print("x_diff_abs: ", x_diff_abs)
        # print("y_diff_abs: ", y_diff_abs)
        # print("z_diff_abs: ", z_diff_abs)
        if x_diff_abs < 3 and y_diff_abs < 3:
                self.GRASP_STATUS = 1
        if not self.hori_pickup: # only relevant for veritical pickup to avoid collision with surface
            # stop if current z is less than threshold
            if  (z_diff_abs < 1) or (z < z_val_min): # grasping and z being out of bounds
                print("Stopping topdown pickup bc of z diff")
                self.stop_motion()
        else:
            if (x_diff_abs < 1) and (y_diff_abs < 1): # case of grasping
                print("Trigger grasping in hori pickup - within xy bounds")
                self.stop_motion()
        
    def get_eef_pose_m(self):
        _, eef_pos_mm = self.arm.get_position(is_radian=True)
        eef_pos_m = [eef_pos_mm[0]*0.001, eef_pos_mm[1]*0.001, eef_pos_mm[2]*0.001, eef_pos_mm[3], eef_pos_mm[4], eef_pos_mm[5]]
        return eef_pos_m

    def wrap90_closest0(self, angle):
        """
        Convert an angle to the representation in [-90, 90] that is closest to zero.
        
        Args:
            angle: Input angle in degrees
            
        Returns:
            The equivalent angle within [-90, 90] that is closest to zero
        """
        closest_angle = None
        min_distance = float('inf')
        
        # Try different equivalent angles (multiples of 90°)
        for offset in range(-8, 9):  # Sufficient range to find the closest
            temp = angle + offset * 90
            
            # Ensure it's in [-90, 90] range
            normalized = temp
            while normalized > 90: 
                normalized -= 180
            while normalized < -90:
                normalized += 180
            
            # Check if this is closer to zero
            distance = abs(normalized)
            if distance < min_distance:
                min_distance = distance
                closest_angle = normalized
        
        return closest_angle

    def grasp(self, data):
        # print("data: ", data)
        # print("Checking conditions:")  # Add this line
        # print(f"self.alive: {self.alive}")  # Add this line
        # print(f"self.is_ready: {self.is_ready}")  # Add this line
        # print(f"self.SERVO: {self.SERVO}")  # Add this line

        if not self.alive or not self.is_ready or not self.SERVO:
            # print("data in condition: ",data)
            return

        d = list(data)

        if len(d) <= 6: # 7th element is the eef pose
            # PBVS Method.
            print("getting pose after grasp img is calculated")
            euler_base_to_eef = self.get_eef_pose_m()
            # print("grasp data eef: ", euler_base_to_eef)
        else:
            # print("grasp data eef: ", d)
            euler_base_to_eef = d[6]
            # print("grasp data eef 6: ", euler_base_to_eef)
        print("d_raw: ", d) # ang in radians
        # print([d[0], d[1], d[2], 0, 0, -d[3]])
        # if d[2] > 0.35:  # Min effective range of the oakdpro.
        print(math.cos(math.radians(15)))
        # cut off detection loop for large or small values 
        if d[2] > self.ggcnn_cutoff_dist and d[2] < self.max_arm_ctrl_dist and self.dist_valid:  # Min effective range of the realsense.
            if not self.hori_pickup:
                gp = [d[0], d[1], d[2], 0, 0, -d[3]] # xyz00(angle of grasp) in meter
            else:
                # Transform grasp point
                
                gp = [d[2]*math.cos(math.radians(15)), -d[1], -d[0]*math.cos(math.radians(15)), 0, 0, 0]  # For horizontal grasping
                # gp = [d[2], -d[0], d[1], 0, 0, 0]  # For horizontal grasping
            
            print("gp", gp)
            # Calculate Pose of Grasp in Robot Base Link Frame
            # Average over a few predicted poses to help combat noise.
            # print("gp", gp)
            e2mat1 = euler2mat(euler_base_to_eef)
            e2mat2 = euler2mat(self.euler_eef_to_color_opt) 

            mat_depthOpt_in_base = euler2mat(euler_base_to_eef) * euler2mat(self.euler_eef_to_color_opt) * euler2mat(self.euler_color_to_depth_opt)
            
            print("euler_base_to_eef matrix:", euler2mat(euler_base_to_eef))
            print("euler_eef_to_color_opt matrix:", euler2mat(self.euler_eef_to_color_opt))
            # print("euler_color_to_depth_opt matrix:", euler2mat(self.euler_color_to_depth_opt))
            print("Combined matrix:", mat_depthOpt_in_base)

            gp_base = convert_pose(gp, mat_depthOpt_in_base)
            print("gp_base regular: ", gp_base)

            gp_base = np.zeros(6)
            # Add the arrays
            # e2mat2_2 =  np.array(e2mat2)
            # print(e2mat2_2[3][2])
            # e2mat2_2[3][2] = d[2]*math.sin(math.radians(15))
            # print(e2mat2_2)
            gp_base[0:3] = e2mat1[0:3, 3].flatten() + e2mat2[0:3, 3].flatten() + np.array(gp[:3])  
            # gp_base = np.concatenate([gp_base, np.array([0, 0, 0])])

            print("gp_base_add: ", gp_base)

            gp_base[3] = d[3]
            if not self.hori_pickup:
                if gp_base[5] < -np.pi:
                    gp_base[5] += np.pi
                elif gp_base[5] > np.pi:
                    gp_base[5] -= np.pi
                
                av = self.pose_averager.update(np.array([gp_base[0], gp_base[1], gp_base[2], gp_base[5]]))
            else:
                # print("Original angle: ", math.degrees(gp_base[3]))
                # Normalize using modulo to get equivalent angle in -180° to +180° range
                gp_base[3] = ((gp_base[3] + np.pi) % (2 * np.pi)) - np.pi
                # print("After first normalization: ", math.degrees(gp_base[3]))
                # if gp_base[3] < -np.pi/2:
                #     gp_base[3] = -np.pi/2
                # elif gp_base[3] > np.pi/2:
                #     gp_base[3] = np.pi/2
                # print("Final normalized angle: ", math.degrees(gp_base[3]))

                # print("Updating pose_averager with:", [gp_base[0], gp_base[1], gp_base[2], gp_base[3]])
                # old_av = self.pose_averager.evaluate() if hasattr(self.pose_averager, 'evaluate') else None
                # print("Pose averager before update:", old_av)
                av = self.pose_averager.update(np.array([gp_base[0], gp_base[1], gp_base[2], gp_base[3]]))
        else:
            self.dist_valid = False # stop reading from the camera
            # gp_base = geometry_msgs.msg.Pose()
            gp_base = [0]*6
            av = self.pose_averager.evaluate()
            print("av during bad distance: ", av)
            
        # print("Pose averager after update:", av)
        # print("av: ", av)

        av = [round(num,6) for num in av] # av is in m
        print("av: ", av)
        # Process initial angle based on pickup type
        ang = av[3] - np.pi/2

        # Get current angle of the robot (in radians)
        if not self.hori_pickup:
            current_angle_rad = math.radians(self.CURR_POS[5])
        else: # use only final axes motion for hori pickup
            _, cur_j_pos = self.arm.get_servo_angle(is_radian=False)
            current_angle_rad = math.radians(cur_j_pos[5])
        
        # print("Current angle (radians): ", current_angle_rad)
        # print("Current angle (degrees): ", self.CURR_POS[5] if not self.hori_pickup else self.CURR_POS[3])

        # print("Initial angle (radians): ", ang)
        # print("Initial angle (degrees): ", math.degrees(ang))

        # # Normalize the angle to avoid full rotations
        while ang < -np.pi:
            ang += 2*np.pi
        while ang > np.pi:
            ang -= 2*np.pi

        # print("Normalized angle (radians): ", ang)
        # print("Normalized angle (degrees): ", math.degrees(ang))

        # Get representation closest to zero
        zero_closest_ang_deg = self.wrap90_closest0(math.degrees(ang))
        
        # Now find the best representation for minimal robot movement
        # Create options that are equivalent to zero-closest representation
        options = []
        for offset in range(-4, 5):
            temp = zero_closest_ang_deg + offset * 180  # Only need to check 180° offsets now
            temp_rad = math.radians(temp)
            
            # Calculate rotation distance, handle wrap-around
            dist = abs(temp_rad - current_angle_rad)
            if dist > np.pi:
                dist = 2*np.pi - dist
                
            options.append((temp_rad, dist))

        # Sort by distance and pick the closest one
        options.sort(key=lambda x: x[1])
        final_ang = options[0][0]

        # Set goal position based on pickup orientation
        if not self.hori_pickup:  # add offset to z dimension for vertical pickup
            GOAL_POS = [av[0] * 1000, 
                        av[1] * 1000, 
                        av[2] * 1000 + self.gripper_z_mm,
                        180, 0, math.degrees(final_ang)]
            if abs(GOAL_POS[2]) < self.gripper_z_mm + 10:
                return
        else:  # add offset to x dimension for horizontal pickup
            # Calculate the desired roll angle
            desired_roll = final_ang
            initial_roll = math.radians(self.init_j_pose[3]) # Starting j6 angle

            current_roll = current_angle_rad
            # print("desired_roll: ", desired_roll)
            # print("initial_roll: ", initial_roll)
            # Calculate the angular difference, accounting for wrap-around
            # This gives us the smallest angle between the two positions, regardless of direction
            angle_diff = math.atan2(math.sin(desired_roll - current_roll), 
                                math.cos(desired_roll - current_roll))
                                
            # print(f"Smallest angle difference: {math.degrees(angle_diff):.1f} degrees")

            # Choose the angle that requires the smallest movement
            final_roll = current_roll + angle_diff

            # Check if the movement is within allowed limits from initial position
            total_deviation = math.atan2(math.sin(final_roll - initial_roll),
                                    math.cos(final_roll - initial_roll))

            # Limit maximum deviation from initial position if needed
            max_allowed_deviation = math.radians(90)  # 90 degrees in radians
            if abs(total_deviation) > max_allowed_deviation:
                # Limit the deviation to max_allowed_deviation in the appropriate direction
                final_roll = initial_roll + math.copysign(max_allowed_deviation, total_deviation)
            
            final_roll = math.degrees(final_roll)
            # print(f"final_roll (deg): {final_roll:.1f}")
            
            roll_diff = round(abs(math.degrees(current_roll) - final_roll),3) # j6 travel desired
            
            # Create offset vector in the direction of the grasp
            offset_x = round(self.gripper_z_mm * math.cos(self.CURR_POS[5]),3)
            offset_y = round(self.gripper_z_mm * math.sin(self.CURR_POS[5]),3)
            # print("offset_x: ", offset_x)
            # print("offset_y: ", offset_y)
            # Apply the offset to the position
            GOAL_POS = [
                # av[0] * 1000 + offset_x,  # X with directional offset
                # av[1] * 1000 + offset_y,  # Y with directional offset
                av[0] * 1000 + self.gripper_z_mm,  # X with directional offset
                av[1] * 1000,  # Y with directional offset
                av[2] * 1000,             # Z unchanged
                # math.degrees(final_roll),  # Roll angle
                self.CURR_POS[3],
                self.CURR_POS[4],         # Pitch unchanged
                self.CURR_POS[5]          # Yaw unchanged
            ]
        # always keep arm above/at min range
        GOAL_POS[0] = max(self.grasping_range[0], GOAL_POS[0])
        GOAL_POS[1] = max(self.grasping_range[2], GOAL_POS[1])
        GOAL_POS[2] = max(self.grasping_min_z, GOAL_POS[2])
        self.last_grasp_time = time.monotonic()
        print("Before control loop")
        print("GOAL_POS", GOAL_POS)
        print("CUR_POS", self.CURR_POS)
        if not self.arm_resetting and self.arm.error_code == 0: # only go to pose if arm is not resetting
            # self.GOAL_POS = GOAL_POS
            # self.arm.set_position(*self.GOAL_POS, speed=50, mvacc=500, wait=False)
            if self.GRASP_STATUS == 0:
                self.GOAL_POS = GOAL_POS
                # _,GOAL_POS_J = self.arm.get_inverse_kinematics(self.GOAL_POS, return_is_radian=False)
                # GOAL_POS_J = np.round(np.array(GOAL_POS_J[:-1],dtype=np.float32),2)
                # print("GOAL_POS_J", GOAL_POS_J)
                # self.arm.set_mode(0)
                # self.arm.set_state(0)
                # self.arm.set_servo_angle(angle=GOAL_POS_J, speed=25, mvacc=1000, wait=False) 
                # align the robot before approaching in z

                if GOAL_POS[0] != self.release_xyz[0]:
                    # xy_bin = abs(self.GOAL_POS[0] - self.CURR_POS[0]) > 20 or abs(self.GOAL_POS[1] - self.CURR_POS[1]) > 20 # mm
                    y_bin = abs(self.GOAL_POS[1] - self.CURR_POS[1]) > 5
                    # print("xy_bin", xy_bin)
                    # ang_bin = (abs(self.GOAL_POS[3] - self.CURR_POS[3]) > 5) or (abs(self.GOAL_POS[4] - self.CURR_POS[4])  > 5) or (abs(self.GOAL_POS[5] - self.CURR_POS[5]) > 5) # deg
                    # print("ang_bin", ang_bin)
                    # print(self.CURR_POS)
                    # print(self.GOAL_POS)
                    # print((-self.CURR_POS[2]+self.GOAL_POS[2])*0.5+self.CURR_POS[2])
                    if not self.hori_pickup:
                            self.arm.set_position(x=self.GOAL_POS[0], y=self.GOAL_POS[1], z = (-self.CURR_POS[2]+self.GOAL_POS[2])*0.5+self.CURR_POS[2],
                                                roll=self.GOAL_POS[3], pitch=self.GOAL_POS[4], yaw=self.GOAL_POS[5], 
                                                speed=75, mvacc=750, wait=False)
                    else: # adjust angle first and approach after
                        # print("current_roll, desired_roll, roll_diff ", math.degrees(current_roll), final_roll, roll_diff)
                        roll_tol = 50000 # degrees
                        # change modes only once
                        if abs(roll_diff) > roll_tol:
                            if self.xy_ctrl:
                                self.roll_ctrl = True
                                self.xy_ctrl = False
                                # continue to change mode until success
                                # self.arm.set_state(3) # pause motion while changing j6
                                mode_change_bool = 1
                                while mode_change_bool != 0:
                                    mode_change_bool = self.arm.set_mode(1)
                                time.sleep(1)
                                self.arm.set_state(0)
                        else:
                            self.xy_ctrl = True
                            if self.roll_ctrl:
                                self.arm.set_mode(7)
                                time.sleep(0.5)
                                self.arm.set_state(0)
                                self.roll_ctrl = False

                        if abs(roll_diff) > roll_tol:
                            print("aligning j6")
                            # self.arm.set_position(z=self.GOAL_POS[2], y = (-self.CURR_POS[1]+self.GOAL_POS[1])*0.75+self.CURR_POS[1], 
                            #                     x = (-self.CURR_POS[0]+self.GOAL_POS[0])*0.75+self.CURR_POS[0],
                            #                     roll=self.GOAL_POS[3], pitch=self.GOAL_POS[4], yaw=self.GOAL_POS[5], 
                            #                     speed=75, mvacc=750, wait=False)
                            # self.arm.set_position(roll=self.GOAL_POS[3], wait = True)
                            # self.arm.set_position(z=self.CURR_POS[2], y = self.CURR_POS[1], 
                            #                     x = self.CURR_POS[0],
                            #                     roll=self.GOAL_POS[3], pitch=self.CURR_POS[4], yaw=self.CURR_POS[5], 
                            #                     speed=75, mvacc=750, wait=False)
                            _, curr_j_pos = self.arm.get_servo_angle(is_radian=False)
                            goal_pos = curr_j_pos
                            goal_pos[5] = final_roll 
                            # print("CUR_POS", self.CURR_POS)
                            # print("curr_j_pos: ", curr_j_pos)
                            # print("goal_pos: ", goal_pos)

                            self.arm.set_servo_angle_j(angles=goal_pos, speed=50, mvacc=1000, wait=True)
                            # self.arm.set_servo_angle(angle=goal_pos, speed=50, mvacc=1000, wait=True)
                            
                            # print("angle alignment executing....") 
                        else:
                            print("After angle alignment, during control loop")
                            print("GOAL_POS", GOAL_POS)
                            print("CUR_POS", self.CURR_POS)
                            # align y first and then go to x
                            if y_bin:
                                self.arm.set_position(x=self.CURR_POS[0], y=self.GOAL_POS[1], z = self.GOAL_POS[2],
                                                roll=self.GOAL_POS[3], pitch=self.GOAL_POS[4], yaw=self.GOAL_POS[5], 
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
                print("Final approach to grasping")
                self.GOAL_POS = GOAL_POS
                self.arm.set_position(*self.GOAL_POS, speed=50, acc=1000, wait=False)
