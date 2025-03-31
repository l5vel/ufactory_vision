# claude generated, buggy waypoints - WPI? 
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


class GraspPos:
    def __init__(self, threshold_x=20, threshold_y=20):
        self.threshold_x = threshold_x
        self.threshold_y = threshold_y
        self.grasp_pos_a = [0] * 6
        self.grasp_pos_b = [0] * 6
        self.grasp_pos_c = [0] * 6
        self.step = 0
    
    def set_step(self, step):
        self.step = step
    
    def set_pos_a(self, pos):
        for i in range(6):
            self.grasp_pos_a[i] = pos[i]
    
    def set_pos_b(self, pos):
        for i in range(6):
            self.grasp_pos_b[i] = pos[i]
    
    def set_pos_c(self, pos):
        for i in range(6):
            self.grasp_pos_c[i] = pos[i]
    
    def update_pos_a_from_b(self):
        for i in range(6):
            self.grasp_pos_a[i] = self.grasp_pos_b[i]
    
    def update_pos_a_from_bc(self):
        for i in range(6):
            self.grasp_pos_a[i] = self.grasp_pos_c[i]
        self.grasp_pos_a[2] = self.grasp_pos_b[2]

    def check_ab(self):
        if abs(self.grasp_pos_a[0] - self.grasp_pos_b[0]) > self.threshold_x or abs(self.grasp_pos_a[1] - self.grasp_pos_b[1]) > self.threshold_y:
            return True
        return False

    def check_bc(self):
        if abs(self.grasp_pos_b[0] - self.grasp_pos_c[0]) > self.threshold_x or abs(self.grasp_pos_b[1] - self.grasp_pos_c[1]) > self.threshold_y:
            return True
        return False


# Added new class for collision modeling and avoidance
class CollisionModel:
    def __init__(self, resolution=10, safety_margin=20):
        # Resolution in mm for the occupancy grid
        self.resolution = resolution
        # Safety margin in mm
        self.safety_margin = safety_margin
        # Occupancy grid dimensions (x, y) in grid cells
        self.grid_dims = (100, 100)  # 1m x 1m area
        # Occupancy grid origin in robot base frame (x, y, z) in mm
        self.grid_origin = [200, -500, 0]
        # Initialize occupancy grid (2.5D height map)
        self.occupancy_grid = np.zeros(self.grid_dims)
        # Flag to indicate if the grid is initialized
        self.is_initialized = False
        # Lock for thread safety
        self.lock = threading.Lock()
        
    def update_from_depth_image(self, depth_image, camera_intrinsics, camera_extrinsics):
        """
        Update the collision model using a depth image
        
        Args:
            depth_image: Depth image (HxW) in meters
            camera_intrinsics: Camera intrinsic parameters
            camera_extrinsics: Camera extrinsic parameters (transform from camera to robot base)
        """
        with self.lock:
            # Reset occupancy grid
            self.occupancy_grid = np.zeros(self.grid_dims)
            
            # Get image dimensions
            height, width = depth_image.shape
            
            # Extract camera parameters
            fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
            cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
            
            # For each pixel in the depth image
            for v in range(0, height, 4):  # Downsample for efficiency
                for u in range(0, width, 4):
                    # Get depth value
                    z = depth_image[v, u]
                    
                    # Skip invalid depth values
                    if z <= 0 or z > 2.0:
                        continue
                    
                    # Convert pixel to 3D point in camera frame
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    
                    # Convert to robot base frame
                    point_cam = np.array([x, y, z, 1.0])
                    point_base = camera_extrinsics @ point_cam
                    
                    # Convert to grid coordinates
                    grid_x = int((point_base[0] * 1000 - self.grid_origin[0]) / self.resolution)
                    grid_y = int((point_base[1] * 1000 - self.grid_origin[1]) / self.resolution)
                    
                    # Skip if outside grid
                    if grid_x < 0 or grid_x >= self.grid_dims[0] or grid_y < 0 or grid_y >= self.grid_dims[1]:
                        continue
                    
                    # Update height map (maximum height at each grid cell)
                    height_mm = point_base[2] * 1000
                    self.occupancy_grid[grid_y, grid_x] = max(self.occupancy_grid[grid_y, grid_x], height_mm)
            
            # Apply safety margin
            self.occupancy_grid = np.array(self.occupancy_grid)
            self.is_initialized = True
            
    def update_from_point_cloud(self, points_3d, transform_matrix=None):
        """
        Update the collision model using a point cloud
        
        Args:
            points_3d: Nx3 array of 3D points in meters
            transform_matrix: 4x4 transformation matrix from point cloud frame to robot base frame
        """
        with self.lock:
            # Reset occupancy grid
            self.occupancy_grid = np.zeros(self.grid_dims)
            
            # Transform points to robot base frame if needed
            if transform_matrix is not None:
                points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
                points_base = (transform_matrix @ points_homogeneous.T).T[:, :3]
            else:
                points_base = points_3d
            
            # Convert to grid coordinates and update height map
            for point in points_base:
                # Convert to mm
                x_mm, y_mm, z_mm = point * 1000
                
                # Convert to grid coordinates
                grid_x = int((x_mm - self.grid_origin[0]) / self.resolution)
                grid_y = int((y_mm - self.grid_origin[1]) / self.resolution)
                
                # Skip if outside grid
                if grid_x < 0 or grid_x >= self.grid_dims[0] or grid_y < 0 or grid_y >= self.grid_dims[1]:
                    continue
                
                # Update height map (maximum height at each grid cell)
                self.occupancy_grid[grid_y, grid_x] = max(self.occupancy_grid[grid_y, grid_x], z_mm)
            
            # Apply safety margin
            self.occupancy_grid = np.array(self.occupancy_grid)
            self.is_initialized = True
    
    def check_collision(self, position, radius=50):
        """
        Check if a position is in collision with the occupancy grid
        
        Args:
            position: [x, y, z] in mm
            radius: Safety radius in mm
            
        Returns:
            bool: True if in collision, False otherwise
        """
        if not self.is_initialized:
            return False
        
        with self.lock:
            # Convert position to grid coordinates
            grid_x = int((position[0] - self.grid_origin[0]) / self.resolution)
            grid_y = int((position[1] - self.grid_origin[1]) / self.resolution)
            
            # Check if position is within grid bounds
            if grid_x < 0 or grid_x >= self.grid_dims[0] or grid_y < 0 or grid_y >= self.grid_dims[1]:
                return False
            
            # Check collision with height map
            radius_cells = int(radius / self.resolution)
            
            # Bounds check for the region of interest
            min_x = max(0, grid_x - radius_cells)
            max_x = min(self.grid_dims[0] - 1, grid_x + radius_cells)
            min_y = max(0, grid_y - radius_cells)
            max_y = min(self.grid_dims[1] - 1, grid_y + radius_cells)
            
            # Check if any cell in the region has a height greater than position[2]
            region = self.occupancy_grid[min_y:max_y+1, min_x:max_x+1]
            return np.any(region > position[2])
    
    def find_safe_path(self, start_pos, goal_pos, max_iterations=100):
        """
        Find a safe path from start to goal position
        
        Args:
            start_pos: [x, y, z] in mm
            goal_pos: [x, y, z] in mm
            max_iterations: Maximum number of iterations
            
        Returns:
            list: List of waypoints [[x, y, z], ...] in mm
        """
        if not self.is_initialized:
            return [start_pos, goal_pos]
        
        with self.lock:
            # Simplified RRT-like algorithm
            waypoints = [np.array(start_pos)]
            current_pos = np.array(start_pos)
            goal_pos = np.array(goal_pos)
            
            for _ in range(max_iterations):
                # Vector from current to goal
                direction = goal_pos - current_pos
                distance = np.linalg.norm(direction)
                
                # If close enough to goal, add goal and return
                if distance < 10:  # 10mm threshold
                    waypoints.append(goal_pos)
                    return waypoints
                
                # Normalize direction and create a step
                step_size = min(50, distance)  # 50mm steps
                direction = direction / distance * step_size
                next_pos = current_pos + direction
                
                # Check for collision
                if self.check_collision(next_pos):
                    # Try to find a safe step in a different direction
                    found_safe_step = False
                    
                    # Try different directions with increasing height
                    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                        for height_offset in [0, 20, 50, 100]:
                            test_direction = np.array([
                                direction[0] * np.cos(angle) - direction[1] * np.sin(angle),
                                direction[0] * np.sin(angle) + direction[1] * np.cos(angle),
                                direction[2] + height_offset
                            ])
                            test_pos = current_pos + test_direction
                            
                            if not self.check_collision(test_pos):
                                next_pos = test_pos
                                found_safe_step = True
                                break
                        
                        if found_safe_step:
                            break
                    
                    # If no safe step found, try going up
                    if not found_safe_step:
                        next_pos = current_pos + np.array([0, 0, step_size])
                
                # Add waypoint and update current position
                waypoints.append(next_pos)
                current_pos = next_pos
            
            # If max iterations reached, try direct path to goal
            if not self.check_collision(goal_pos):
                waypoints.append(goal_pos)
            
            return waypoints


class TrajectoryPlanner:
    def __init__(self, collision_model, max_speed=100, max_acc=1000):
        self.collision_model = collision_model
        self.max_speed = max_speed  # mm/s
        self.max_acc = max_acc  # mm/s^2
        
    def plan_trajectory(self, start_pos, goal_pos):
        """
        Plan a safe trajectory from start to goal position
        
        Args:
            start_pos: [x, y, z, roll, pitch, yaw] in mm and degrees
            goal_pos: [x, y, z, roll, pitch, yaw] in mm and degrees
            
        Returns:
            list: List of waypoints [[x, y, z, roll, pitch, yaw], ...] in mm and degrees
        """
        # Find a safe path for position (x, y, z)
        start_xyz = start_pos[:3]
        goal_xyz = goal_pos[:3]
        
        # Get safe path
        safe_path = self.collision_model.find_safe_path(start_xyz, goal_xyz)
        
        # Convert path to full 6-DOF waypoints
        waypoints = []
        
        # Interpolate orientation
        start_rpy = np.array(start_pos[3:])
        goal_rpy = np.array(goal_pos[3:])
        path_length = len(safe_path)
        
        for i, xyz in enumerate(safe_path):
            # Linear interpolation for orientation
            alpha = i / (path_length - 1) if path_length > 1 else 1
            rpy = start_rpy * (1 - alpha) + goal_rpy * alpha
            
            # Create full 6-DOF waypoint
            waypoint = np.concatenate([xyz, rpy]).tolist()
            waypoints.append(waypoint)
        
        return waypoints
        
    def execute_trajectory(self, arm, waypoints, speed=None, acc=None):
        """
        Execute a trajectory on the robot arm
        
        Args:
            arm: XArmAPI instance
            waypoints: List of waypoints [[x, y, z, roll, pitch, yaw], ...] in mm and degrees
            speed: Speed for motion (mm/s)
            acc: Acceleration for motion (mm/s^2)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if speed is None:
            speed = self.max_speed
        if acc is None:
            acc = self.max_acc
            
        # Execute waypoints
        for i, waypoint in enumerate(waypoints):
            # Last waypoint should wait until the motion is complete
            wait = (i == len(waypoints) - 1)
            
            # Move to waypoint
            code = arm.set_position(
                x=waypoint[0], y=waypoint[1], z=waypoint[2],
                roll=waypoint[3], pitch=waypoint[4], yaw=waypoint[5],
                speed=speed, acc=acc, wait=wait
            )
            
            if code != 0:
                return False
            
        return True


class RobotGrasp(object):    
    CURR_POS = [300, 0, 350, 180, 0, 0]
    GOAL_POS = [0, 0, 0, 0, 0, 0]

    SERVO = True
    GRASP_STATUS = 0

    def __init__(self, robot_ip, ggcnn_cmd_que, euler_eef_to_color_opt, euler_color_to_depth_opt, grasping_range, detect_xyz, gripper_z_mm, release_xyz, grasping_min_z, use_init_pos=False):
        self.arm = XArmAPI(robot_ip, report_type='real')
        self.ggcnn_cmd_que = ggcnn_cmd_que
        self.euler_eef_to_color_opt = euler_eef_to_color_opt
        self.euler_color_to_depth_opt = euler_color_to_depth_opt
        self.grasping_range = grasping_range
        self.use_init_pos = use_init_pos
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
        self.grasp_pos = GraspPos()
        
        # Initialize collision model and trajectory planner
        self.collision_model = CollisionModel(resolution=10, safety_margin=20)
        self.trajectory_planner = TrajectoryPlanner(self.collision_model)
        self.latest_depth_image = None
        self.camera_intrinsics = None
        self.camera_extrinsics = None
        
        self.pos_t = threading.Thread(target=self.update_pos_loop, daemon=True)
        self.pos_t.start()
        self.ggcnn_t = threading.Thread(target=self.handle_ggcnn_loop, daemon=True)
        self.ggcnn_t.start()
        self.check_t = threading.Thread(target=self.check_loop, daemon=True)
        self.check_t.start()
    
    def is_alive(self):
        return self.alive
    
    def update_depth_data(self, depth_image, camera_intrinsics, camera_extrinsics):
        """
        Update depth data for collision avoidance
        
        Args:
            depth_image: Depth image (HxW) in meters
            camera_intrinsics: Camera intrinsic parameters
            camera_extrinsics: Camera extrinsic parameters (transform from camera to robot base)
        """
        self.latest_depth_image = depth_image
        self.camera_intrinsics = camera_intrinsics
        self.camera_extrinsics = camera_extrinsics
        
        # Update collision model
        self.collision_model.update_from_depth_image(depth_image, camera_intrinsics, camera_extrinsics)
    
    def handle_ggcnn_loop(self):
        while self.arm.connected and self.alive:
            cmd = self.ggcnn_cmd_que.get()
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
        self.arm.set_servo_angle(angle=self.init_j_pose, speed=50, mvacc=1000, wait=True) # going to grasp position but in joint space to avoid IK errors
        if not(self.arm.get_gripper_position()[1]) < 0: # only go to recepticle if gripped something
            # Plan safe trajectory to the release position
            curr_pos = self.CURR_POS.copy()
            release_pos = [self.release_xyz[0], self.release_xyz[1], curr_pos[2], 180, 0, 0]
            
            # Plan and execute safe path to above release position
            waypoints = self.trajectory_planner.plan_trajectory(curr_pos, release_pos)
            self.trajectory_planner.execute_trajectory(self.arm, waypoints)
            
            # Move down to final release position
            final_release_pos = [self.release_xyz[0], self.release_xyz[1], self.release_xyz[2], 180, 0, 0]
            self.arm.set_position(z=final_release_pos[2], speed=100, wait=True)
        
        # Open Fingers
        self.arm.set_gripper_position(850, wait=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_servo_angle(angle=self.init_j_pose, speed=50, mvacc=1000, wait=True) # going to grasp position but in joint space to avoid IK errors

        self.pose_averager.reset()
        self.arm.set_mode(7)
        self.arm.set_state(0)
        time.sleep(2)

        self.SERVO = True
        self.last_grasp_time = time.monotonic()

    def update_pos_loop(self):
        self.arm.motion_enable(True)
        self.arm.clean_error()
        self.arm.set_mode(0)
        self.arm.set_state(0)
        _, init_pos = tuple(self.arm.get_initial_point())
        self.init_j_pose = init_pos
        self.arm.set_servo_angle(angle=init_pos,wait=True,is_radian=False)
        time.sleep(0.5)
        _,init_pose = self.arm.get_position(is_radian=True)
        self.init_pose = np.array(init_pose,dtype=np.float64)
        self.detect_xyz = ([init_pose[0],init_pose[1],init_pose[2]])
        self.detect_rpy = ([init_pose[3],init_pose[4],init_pose[5]])
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_position(850)
        time.sleep(0.5)

        self.SERVO = True

        self.arm.set_state(0)
        time.sleep(0.5)

        self.is_ready = True
        self.grasp_pos.set_step(1)

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
        print("arm error code", self.arm.error_code)
        while self.arm.connected and self.arm.error_code == 0:
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
        if (self.last_grasp_time != 0 and (time.monotonic() - self.last_grasp_time) > 5)  or x < self.grasping_range[0] or x > self.grasping_range[1] or y < self.grasping_range[2] or y > self.grasping_range[3]:
            if (time.monotonic() - self.last_grasp_time) > 5 \
                and abs(x-self.detect_xyz[0]) < 2 and abs(y-self.detect_xyz[1]) < 2 and abs(z-self.detect_xyz[2]) < 2 \
                and abs(abs(roll)-180) < 2 and abs(pitch) < 2 and abs(yaw) < 2:
                self.last_grasp_time = time.monotonic()
                return
            print('*****', self.CURR_POS, self.last_grasp_time)
            self.SERVO = False
            self.arm.set_state(4)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(1)
            
            # Plan a safe path back to the detection position
            curr_pos = self.CURR_POS.copy()
            detect_pos = [self.detect_xyz[0], self.detect_xyz[1], self.detect_xyz[2], 180, 0, 0]
            
            # First move up to a safe height
            safe_height = max(curr_pos[2] + 50, 300)
            self.arm.set_position(z=safe_height, speed=100, wait=True)
            
            # Then plan trajectory to detection position
            curr_pos[2] = safe_height
            waypoints = self.trajectory_planner.plan_trajectory(curr_pos, detect_pos)
            self.trajectory_planner.execute_trajectory(self.arm, waypoints)
            
            time.sleep(0.25)
            self.pose_averager.reset()
            self.arm.set_state(0)
            time.sleep(1)

            self.SERVO = True
            self.last_grasp_time = time.monotonic()
            self.grasp_pos.set_step(1)
            return
        
        if self.grasp_pos.step > 2 and self.grasp_pos.step < 3:
            pos = self.grasp_pos.grasp_pos_a
            if abs(x-pos[0]+50) < 2 and abs(y-pos[1]) < 2:
                time.sleep(1)
                self.grasp_pos.set_step(3)
        if self.grasp_pos.step > 5 and self.grasp_pos.step < 6:
            pos = self.grasp_pos.grasp_pos_b
            pos_z = max(pos[2], 300)
            if abs(x-pos[0]+50) < 2 and abs(y-pos[1]) < 2 and abs(z-pos_z) < 2:
                time.sleep(1)
                self.grasp_pos.set_step(6)
        if self.grasp_pos.step == 4:
            print('===========step4=============')
            print(self.grasp_pos.grasp_pos_a)
            print(self.grasp_pos.grasp_pos_b)
            
            if self.grasp_pos.check_ab():
                self.grasp_pos.update_pos_a_from_b()
                self.grasp_pos.set_step(2)
            else:
                self.grasp_pos.set_step(5)
        elif self.grasp_pos.step == 7:
            print('===========step7==============')
            print(self.grasp_pos.grasp_pos_b)
            print(self.grasp_pos.grasp_pos_c)
            if self.grasp_pos.check_bc():
                self.grasp_pos.update_pos_a_from_bc()
                self.grasp_pos.set_step(2)
            else:
                self.grasp_pos.set_step(8)

        # Stop Conditions.
        if self.grasp_pos.step == 9 or z < self.gripper_z_mm:
            self.stop_motion()
        # Stop Conditions for approach in all directions 
        dist_from_goal =  ((self.CURR_POS[0] - self.GOAL_POS[0])**2 + (self.CURR_POS[1] - self.GOAL_POS[1])**2 + (self.CURR_POS[2] - self.GOAL_POS[2])**2)**0.5
        if dist_from_goal < 1: # ~3**0.5
            self.stop_motion()
        # stop 1mm above the grasp point
        # stop based on the dominant motion direction
        # approach_direction = self.determine_approach_direction()
        # print(approach_direction)
        # # Now you can use approach_direction in your conditional statements
        # if approach_direction in ["top_down"]:
        #     if approach_direction == "top_down":
        #         if z - 1 < self.GOAL_POS[2]:
        #            self.stop_motion()

        # elif approach_direction in ["side_x_positive", "side_x_negative"]:
        #     # Check x threshold with appropriate sign
        #     if approach_direction == "side_x_negative":
        #         if x - 1 < self.GOAL_POS[0]:
        #             self.stop_motion()
        #     else:  # side_x_positive
        #         if x + 1 > self.GOAL_POS[0]:
        #             self.stop_motion()

        # elif approach_direction in ["side_y_positive", "side_y_negative"]:
        #     # Check y threshold with appropriate sign
        #     if approach_direction == "side_y_negative":
        #         if y - 1 < self.GOAL_POS[1]:
        #             self.stop_motion()
        #     else:  # side_y_positive
        #         if y + 1 > self.GOAL_POS[1]:
        #             self.stop_motion()
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
        if d[2] > 0.25:  # Min effective range of the realsense.
            gp = [d[0], d[1], d[2], 0, 0, -d[3]] # xyzrpy in meter
            print("gp", gp)
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
            print("GOAL_POS",GOAL_POS)
            print("CUR_POS", self.CURR_POS)
            # _,GOAL_POS_J = self.arm.get_inverse_kinematics(self.GOAL_POS, return_is_radian=False)
            # GOAL_POS_J = np.round(np.array(GOAL_POS_J[:-1],dtype=np.float64),2)
            # print("GOAL_POS_J", GOAL_POS_J)
            # self.arm.set_mode(0)
            # self.arm.set_state(0)
            # self.arm.set_servo_angle(angle=GOAL_POS_J, speed=25, mvacc=1000, wait=False) 
            self.arm.set_position(x=self.GOAL_POS[0], y=self.GOAL_POS[1], z=self.GOAL_POS[2],
                                  roll=self.GOAL_POS[3], pitch=self.GOAL_POS[4], yaw=self.GOAL_POS[5], 
                                  speed=50, mvacc=2500, wait=False)
            # self.pose_averager.reset()
            # self.pose_averager.update(av)

        elif self.GRASP_STATUS == 1:
            self.GOAL_POS = GOAL_POS
            self.arm.set_position(*self.GOAL_POS, speed=50, acc=1000, wait=False)
