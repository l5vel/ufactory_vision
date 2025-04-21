import sys
import cv2
import time
import math
import numpy as np
import signal
import logging
from queue import Queue
from threading import Thread, Event
from networktables import NetworkTables

from camera.rs_camera import RealSenseCamera
from grasp.ggcnn_torch import TorchGGCNN
from grasp.robot_grasp import RobotGrasp

# Class constants
WIN_NAME = 'RealSense'
WIDTH = 640
HEIGHT = 480

# RGB camera calibration result
EULER_EEF_TO_COLOR_OPT = [0.015, -0.1082, -0.118, 0, math.radians(20), math.pi/2]  # xyzrpy meters_rad #new tilted mount
EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]

# The range of motion of the robot grasping
GRASPING_RANGE = [-50, 680, -450, 400]  # [x_min, x_max, y_min, y_max]

# The distance between the gripping point and the end of the robot arm flange
GRIPPER_Z_MM = -25  # mm

# Release grasping position
RELEASE_XYZ = [400, 400, 270]

# Min z for grasping
GRASPING_MIN_Z = -400

# Initial detection position
DETECT_XYZ = [300, -200, 350]  # [x, y, z] # reset later in the code based on init pose


class NtablesTriggerGrasp:
    """Class to handle robot grasping triggered by NetworkTables events."""
    
    def __init__(self, robot_ip, raspi_ip=None, use_init_pos=True, stop_on_grasp=True, 
                 ggcnn_in_thread=False):
        """Initialize the NtablesTriggerGrasp class.
        
        Args:
            robot_ip (str): IP address of the robot arm
            raspi_ip (str, optional): IP address of the Raspberry Pi for NetworkTables
            use_init_pos (bool, optional): Use initial position set on robot instead of current pose
            stop_on_grasp (bool, optional): Enable grasping in a loop and dropoff goal execution
            ggcnn_in_thread (bool, optional): Run GGCNN in a separate thread
        """
        # Setup logging for NetworkTables
        logging.basicConfig(level=logging.DEBUG)
        
        # Initialize instance variables
        self.robot_ip = robot_ip
        self.use_init_pos = use_init_pos
        self.stop_on_grasp = stop_on_grasp
        self.ggcnn_in_thread = ggcnn_in_thread
        
        # Initialize modules to None
        self.camera = None
        self.ggcnn = None
        self.grasp_module = None
        self.depth_img_que = None
        self.ggcnn_cmd_que = None
        self.monitor_thread = None
        self.stop_event = None
        
        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # Initialize NetworkTables
        if raspi_ip:
            NetworkTables.initialize(server=raspi_ip)
        else:
            NetworkTables.initialize()
        
        # Get Smart Dashboard
        self.sd = NetworkTables.getTable("SmartDashboard")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle signal interrupts."""
        print("Shutting down...")
        self.cleanup()
    
    def initialize_modules(self):
        """Initialize all required modules: camera, GGCNN, and robot grasp."""
        self.depth_img_que = Queue(1)
        self.ggcnn_cmd_que = Queue(1)
        
        # Initialize camera
        self.camera = RealSenseCamera(width=self.WIDTH, height=self.HEIGHT)
        _, depth_intrin = self.camera.get_intrinsics()
        
        # Set camera intrinsics
        self.fx = depth_intrin.fx
        self.fy = depth_intrin.fy
        self.cx = depth_intrin.ppx
        self.cy = depth_intrin.ppy
        
        # Initialize GGCNN
        self.ggcnn = TorchGGCNN(
            self.depth_img_que, 
            self.ggcnn_cmd_que, 
            depth_intrin, 
            width=self.WIDTH, 
            height=self.HEIGHT, 
            run_in_thread=self.ggcnn_in_thread
        )
        
        # Initialize robot grasp module
        self.grasp_module = RobotGrasp(
            self.robot_ip, 
            self.ggcnn_cmd_que, 
            self.EULER_EEF_TO_COLOR_OPT, 
            self.EULER_COLOR_TO_DEPTH_OPT, 
            self.GRASPING_RANGE, 
            self.DETECT_XYZ, 
            self.GRIPPER_Z_MM, 
            self.RELEASE_XYZ, 
            self.GRASPING_MIN_Z, 
            use_init_pos=self.use_init_pos, 
            stop_arm_on_success=self.stop_on_grasp
            on_trigger_mode=True
        )
        
        # Create and start the background thread to monitor navigation status
        self.stop_event = Event()
        self.monitor_thread = Thread(target=self._background_monitor, daemon=True)
        self.monitor_thread.start()
        
        # Allow camera and GGCNN to initialize
        time.sleep(3)
        print("All modules initialized successfully.")
    
    def _background_monitor(self):
        """Background thread to monitor NetworkTables and control the grasping sequence."""
        # Wait for NetworkTables connection
        while not NetworkTables.isConnected() and not self.stop_event.is_set():
            print("Waiting for NetworkTables connection...")
            time.sleep(1)
        
        print("Connected to NetworkTables!")
        grasp_status = 0
        
        while not self.stop_event.is_set():
            try:
                print("Grasp Status:", grasp_status)
                
                if grasp_status == 1:  # successful grasp
                    manip_status = 2
                    self.sd.putNumber("ManStatus", manip_status)
                    time.sleep(1)
                
                nav_status = self.sd.getNumber("NavCmplt", -1)
                dropff_bool = self.sd.getNumber("DropOffPt", -1)  # if dropoff bool is 1, don't execute grasping
                
                if dropff_bool == 1:
                    self.grasp_module.arm.set_gripper_position(850, wait=True)
                    self.cleanup()
                    break
                
                print("Navigation Status:", nav_status)
                
                if nav_status == 0 or nav_status == -1:  # navigation is not complete/NT unreachable
                    manip_status = 0
                    grasp_status = 0
                elif nav_status == 1 and dropff_bool != 1:
                    manip_status = 1
                
                self.sd.putNumber("ManStatus", manip_status)
                
                if manip_status == 1:
                    time.sleep(1)
                    self.perform_grasp()  # will be in this function until grasp complete
                    grasp_status = 1
                
                time.sleep(0.1)  # Add a small delay to avoid excessive CPU usage
            
            except Exception as e:
                print(f"Exception in monitoring thread: {e}")
                time.sleep(1)  # Continue the loop even after an exception
    
    def perform_grasp(self):
        """Execute the grasping sequence."""
        print("Starting Grasp")
        
        if self.camera is None or self.ggcnn is None or self.grasp_module is None:
            print("Modules not initialized. Call initialize_modules() first.")
            return
        
        color_image, depth_image = self.camera.get_images()
        color_shape = color_image.shape
        print("Grasp module status:", self.grasp_module.is_alive())
        
        while self.grasp_module.is_alive():
            color_image, depth_image = self.camera.get_images()
            
            if self.ggcnn_in_thread:
                if not self.depth_img_que.empty():
                    self.depth_img_que.get()
                self.depth_img_que.put([depth_image, self.grasp_module.CURR_POS[2] / 1000.0])
                cv2.imshow(self.WIN_NAME, color_image)  # Display color image
            else:
                data = self.ggcnn.get_grasp_img(
                    depth_image, 
                    self.cx, 
                    self.cy, 
                    self.fx, 
                    self.fy, 
                    self.grasp_module.CURR_POS[2] / 1000.0
                )
                
                if data:
                    if not self.ggcnn_cmd_que.empty():
                        self.ggcnn_cmd_que.get()
                    self.ggcnn_cmd_que.put(data[0])
                    grasp_img = data[1]
                    combined_img = np.zeros((color_shape[0], color_shape[1] + grasp_img.shape[1] + 10, 3), np.uint8)
                    combined_img[:color_shape[0], :color_shape[1]] = color_image
                    combined_img[:grasp_img.shape[0], color_shape[1]+10:color_shape[1]+grasp_img.shape[1]+10] = grasp_img
                    cv2.imshow(self.WIN_NAME, combined_img)  # Display combined image
                else:
                    cv2.imshow(self.WIN_NAME, color_image)  # Display color image
            
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                break
    
    def cleanup(self):
        """Clean up all resources."""
        print("Performing cleanup...")
        
        # Stop the monitoring thread
        if self.stop_event is not None and self.monitor_thread is not None:
            print("Stopping monitoring thread...")
            self.stop_event.set()  # Signal the thread to stop
            if self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2)  # Wait for thread to exit with timeout
                if self.monitor_thread.is_alive():
                    print("Warning: Monitor thread did not exit cleanly")
                else:
                    print("Monitor thread stopped successfully")
            self.monitor_thread = None
            self.stop_event = None
        
        # Stop camera if it exists
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        
        # Shutdown GGCNN thread if running
        if self.ggcnn is not None:
            if hasattr(self.ggcnn, 'stop') and self.ggcnn_in_thread:
                self.ggcnn.stop()
            self.ggcnn = None
        
        # Clear queues if they exist
        if self.depth_img_que is not None:
            while not self.depth_img_que.empty():
                self.depth_img_que.get()
            self.depth_img_que = None
        
        if self.ggcnn_cmd_que is not None:
            while not self.ggcnn_cmd_que.empty():
                self.ggcnn_cmd_que.get()
            self.ggcnn_cmd_que = None
        
        # Stop robot grasp operations if running
        if self.grasp_module is not None and hasattr(self.grasp_module, 'stop'):
            self.grasp_module.stop()
            self.grasp_module = None
        
        # Close opencv windows
        cv2.destroyAllWindows()
        
        # Shutdown network tables
        NetworkTables.shutdown()
        
        print("Cleanup complete.")
    
    def run(self):
        """Main method to run the grasping application."""
        self.initialize_modules()
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting program due to keyboard interrupt")
        finally:
            self.cleanup()


def main():
    """Entry point function."""
    if len(sys.argv) < 2:
        print('Usage: {} <arm_ip> [raspi_ip]'.format(sys.argv[0]))
        return 1
    
    robot_ip = sys.argv[1]
    raspi_ip = sys.argv[2] if len(sys.argv) > 2 else None
    
    grasp_controller = NtablesTriggerGrasp(robot_ip, raspi_ip)
    grasp_controller.run()


if __name__ == '__main__':
    main()