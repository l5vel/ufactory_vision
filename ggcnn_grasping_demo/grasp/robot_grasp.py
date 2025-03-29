import time
import math
import threading
import numpy as np
from .helpers.matrix_funcs import euler2mat, convert_pose, rot_to_rpy
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

    def __init__(self, robot_ip, ggcnn_cmd_que, euler_eef_to_color_opt, euler_color_to_depth_opt, grasping_range, detect_xyz, gripper_z_mm, release_xyz, grasping_min_z):
        self.arm = XArmAPI(robot_ip, report_type='real')
        self.ggcnn_cmd_que = ggcnn_cmd_que
        self.euler_eef_to_color_opt = euler_eef_to_color_opt
        self.euler_color_to_depth_opt = euler_color_to_depth_opt
        self.grasping_range = grasping_range
        self.detect_xyz = detect_xyz
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
    
    def is_alive(self):
        return self.alive
    
    def handle_ggcnn_loop(self):
        while self.arm.connected and self.alive:
            cmd = self.ggcnn_cmd_que.get()
            self.grasp(cmd)

    def update_pos_loop(self):
        self.arm.motion_enable(True)
        self.arm.clean_error()
        self.arm.set_mode(0)
        self.arm.set_state(0)
        _, init_pos = tuple(self.arm.get_initial_point())
        self.arm.set_servo_angle(angle=init_pos,wait=True,is_radian=False)
        time.sleep(0.5)
        _,init_pose = self.arm.get_position()
        self.detect_xyz = ([init_pose[0],init_pose[1],init_pose[2]])
        # self.arm.set_position(z=self.detect_xyz[2], wait=True)
        # self.arm.set_position(x=self.detect_xyz[0], y=self.detect_xyz[1], z=self.detect_xyz[2], roll=180, pitch=0, yaw=0, wait=True)
        # time.sleep(0.5)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_position(850)
        time.sleep(0.5)

        self.SERVO = True

        self.arm.set_mode(7)
        self.arm.set_state(0)
        time.sleep(0.5)

        self.is_ready = True

        while self.arm.connected and self.arm.error_code == 0:
            _, pos = self.arm.get_position()
            self.arm.get_err_warn_code()
            self.CURR_POS = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]
            time.sleep(0.01)
        self.alive = False
        if self.arm.error_code != 0:
            print('ERROR_CODE: {}'.format(self.arm.error_code))
            print('*************** PROGRAM OVER *******************')
    
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
            print("restart triggered")
            self.SERVO = False
            self.arm.set_state(4)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(1)
            # going to grasp position but in joint space to avoid IK errors
            self.arm.arm.set_servo_angle(angle=[-33.4, 7.0, -38.7, -0.1, -45.8, -33.3], speed=8, mvacc=1145, wait=True)
            # self.arm.set_position(z=self.detect_xyz[2], speed=200, wait=True)
            # self.arm.set_position(x=self.detect_xyz[0], y=self.detect_xyz[1], z=self.detect_xyz[2], roll=180, pitch=0, yaw=0, speed=200, wait=True)
            time.sleep(0.25)
            self.pose_averager.reset()
            self.arm.set_mode(7)
            self.arm.set_state(0)
            time.sleep(0.5)
            self.GRASP_STATUS = 0
            self.SERVO = True
            self.last_grasp_time = time.monotonic()
            return

        # if abs(self.CURR_POS[0] - self.GOAL_POS[0]) < 5 and abs(self.CURR_POS[1] - self.GOAL_POS[1]) < 5 and abs(self.CURR_POS[5] - self.GOAL_POS[5]) < 5:
        if abs(self.CURR_POS[0] - self.GOAL_POS[0]) < 5 and abs(self.CURR_POS[1] - self.GOAL_POS[1]) < 5:
            self.GRASP_STATUS = 1

        # Stop Conditions.
        if z < self.gripper_z_mm or z - 1 < self.GOAL_POS[2]:
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
            self.arm.set_position(z=self.detect_xyz[2], speed=200, wait=True)
            if not(self.arm.get_gripper_position()[1]) < 0: # only go to recepticle if gripped something
                self.arm.set_position(x=self.release_xyz[0], y=self.release_xyz[1], roll=180, pitch=0, yaw=0, speed=200, wait=True)
                self.arm.set_position(z=self.release_xyz[2], speed=100, wait=True)
            # time.sleep(3)
            # input('Press Enter to Complete')

            # Open Fingers
            self.arm.set_gripper_position(850, wait=True)
            # time.sleep(5)

            self.arm.set_position(z=self.detect_xyz[2] + 100, speed=100, wait=True)
            self.arm.set_position(x=self.detect_xyz[0], y=self.detect_xyz[1], z=self.detect_xyz[2], roll=180, pitch=0, yaw=0, speed=200, wait=True)

            self.pose_averager.reset()
            self.arm.set_mode(7)
            self.arm.set_state(0)
            time.sleep(2)

            # input('Press Enter to Start')
            self.SERVO = True
            self.last_grasp_time = time.monotonic()

    def get_eef_pose_m(self):
        _, eef_pos_mm = self.arm.get_position(is_radian=True)
        eef_pos_m = [eef_pos_mm[0]*0.001, eef_pos_mm[1]*0.001, eef_pos_mm[2]*0.001, eef_pos_mm[3], eef_pos_mm[4], eef_pos_mm[5]]
        return eef_pos_m

    def grasp(self, data):
        if not self.alive or not self.is_ready or not self.SERVO:
            return

        d = list(data)

        # PBVS Method.
        euler_base_to_eef = self.get_eef_pose_m()
        if d[2] > 0.2:  # Min effective range of the realsense.
            gp = [d[0], d[1], d[2], 0, 0, -d[3]] # xyzrpy in meter
            # Calculate Pose of Grasp in Robot Base Link Frame
            # Average over a few predicted poses to help combat noise.
            # print("gp", gp)
            mat_depthOpt_in_base = euler2mat(euler_base_to_eef) * euler2mat(self.euler_eef_to_color_opt) * euler2mat(self.euler_color_to_depth_opt)
            # print(euler2mat(self.euler_eef_to_color_opt))
            # print("mat_depthOpt_in_base", mat_depthOpt_in_base)
            # print("euler frame", rot_to_rpy(mat_depthOpt_in_base[0:3,0:3]))
            gp_base = convert_pose(gp, mat_depthOpt_in_base)
            # print("gp_base",gp_base)
            if gp_base[5] < -np.pi:
                gp_base[5] += np.pi
            elif gp_base[5] > 0:
                gp_base[5] -= np.pi 

            # Only really care about rotation about z (e[2]).
            # if gp_base[0] * 1000 < self.grasping_range[0] or gp_base[0] * 1000 > self.grasping_range[1] \
            #     or gp_base[1] * 1000 < self.grasping_range[2] or gp_base[1] * 1000 > self.grasping_range[3]:
            #     return
            av = self.pose_averager.update(np.array([gp_base[0], gp_base[1], gp_base[2], gp_base[5]]))

        else:
            # gp_base = geometry_msgs.msg.Pose()
            gp_base = [0]*6
            av = self.pose_averager.evaluate()

        # Average pose in base frame.
        ang = av[3] - np.pi/2  # We don't want to align, we want to grip.
        gp_base = [av[0], av[0], av[0], np.pi, 0, ang]
        GOAL_POS = [av[0] * 1000, av[1] * 1000, av[2] * 1000 + self.gripper_z_mm, 180, 0, math.degrees(ang + np.pi)]
        # print("GOAL_POS", GOAL_POS)
        if abs(GOAL_POS[2]) < self.gripper_z_mm + 10:
            # print('[IG]', GOAL_POS)
            return
        # print(GOAL_POS[2])
        # GOAL_POS[2] = min(abs(euler_base_to_eef[2]-d[2]), euler_base_to_eef[2]) # forced to only pick up from table
        # GOAL_POS[2] = GOAL_POS[2]*1000
        # print(GOAL_POS)
        GOAL_POS[2] = max(GOAL_POS[2],self.grasping_min_z)
        # print(GOAL_POS)
        self.last_grasp_time = time.monotonic()

        # self.GOAL_POS = GOAL_POS
        # self.arm.set_position(*self.GOAL_POS, speed=30, acc=1000, wait=False)
        if self.GRASP_STATUS == 0:
            self.GOAL_POS = GOAL_POS
            self.arm.set_position(x=self.GOAL_POS[0], y=self.GOAL_POS[1], z=self.GOAL_POS[2],
                                  roll=self.GOAL_POS[3], pitch=self.GOAL_POS[4], yaw=self.GOAL_POS[5], 
                                  speed=50, acc=1000, wait=False)
            # self.pose_averager.reset()
            # self.pose_averager.update(av)

        elif self.GRASP_STATUS == 1:
            self.GOAL_POS = GOAL_POS
            self.arm.set_position(*self.GOAL_POS, speed=50, acc=1000, wait=False)
