import sys
import cv2
import time
import numpy as np
from camera.rs_camera import RealSenseCamera
from grasp.ggcnn_torch import TorchGGCNN
from grasp.robot_grasp import RobotGrasp
from queue import Queue
import math

# comm b/w pi and jetson
import signal
from threading import Thread, Event
from networktables import NetworkTables
# To see messages from networktables, you must setup logging
import logging

logging.basicConfig(level=logging.DEBUG)

raspi_ip = '172.16.0.12'
NetworkTables.initialize(server=raspi_ip)
sd = NetworkTables.getTable("PiJetson")
robot_ip = '172.16.0.13' # default

WIN_NAME = 'RealSense'
WIDTH = 640
HEIGHT = 480
## rgb camera calibration result
# EULER_EEF_TO_COLOR_OPT = [0.015, 0.06, -0.15, 0, 0, math.pi/2] # xyzrpy meters_rad #old mount
# rgb camera calibration result
# EULER_EEF_TO_COLOR_OPT = [0.1082, 0.015, -0.125, 0, -math.radians(20), math.pi/2] # xyzrpy meters_rad #new tilted mount
EULER_EEF_TO_COLOR_OPT = [0.015, -0.1082, -0.118, 0, math.radians(20), math.pi/2] # xyzrpy meters_rad #new tilted mount

EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]
GGCNN_IN_THREAD = False

# The range of motion of the robot grasping
# If it exceeds the range, it will return to the initial detection position.
# ### Table mounted
# # GRASPING_RANGE = [180, 600, -200, 200] # [x_min, x_max, y_min, y_max]
# # GRASPING_RANGE = [0, 1000, -500, 600] # [x_min, x_max, y_min, y_max]
# # GRASPING_RANGE = [0, 700, -500, 300] # [x_min, x_max, y_min, y_max]
# # GRASPING_RANGE = [0, 700, -1000,0] # [x_min, x_max, y_min, y_max]
# GRASPING_RANGE = [-500, 700, -1000,100] # [x_min, x_max, y_min, y_max]
# # The distance between the gripping point of the robot grasping and the end of the robot arm flange
# # The value needs to be fine-tuned according to the actual situation.
# GRIPPER_Z_MM = -40 # mm

# # release grasping pos
# RELEASE_XYZ = [400, 400, 270]

# # min z for grasping
# GRASPING_MIN_Z = 0

### Swerve Drive
GRASPING_RANGE = [-50, 680, -450, 400] # [x_min, x_max, y_min, y_max]
# The distance between the gripping point of the robot grasping and the end of the robot arm flange
# The value needs to be fine-tuned according to the actual situation.
GRIPPER_Z_MM = -25 # mm - accounting for the shifted camera

# release grasping pos
RELEASE_XYZ = [400, 350, 400]

# min z for grasping
GRASPING_MIN_Z = -400

#  initial detection position
DETECT_XYZ = [300, -200, 350] # [x, y, z] # reset later in the code based on init pose
# Use initial position set on robot instead of predefined initial detectino position
USE_INIT_POS = True 

def signal_handler(sig, frame):
    print("Shutting down...")
    # Perform cleanup if needed
    NetworkTables.shutdown()
    sys.exit(0)

def background_monitor(stop_event):
    # Add connection status check
    while not NetworkTables.isConnected() and not stop_event.is_set():
        print("Waiting for NetworkTables connection...")
        time.sleep(1)
    
    print("Connected to NetworkTables!")
    grasp_status = 0
    while True:
        try:
            nav_status = sd.getNumber("NavCmplt", -1)
            print("Navigation Status:", nav_status)
            if nav_status == 0 or nav_status == -1:  # navigation is not complete/NT unreachable
                manip_status = 0
            elif nav_status == 1:
                manip_status = 1
            sd.putNumber("ManStatus", manip_status)
            if manip_status == 1:# wait for the pi to shutdown
                time.sleep(1) 
            if manip_status == 1:    
                grasp() # will be in this function until exited
            time.sleep(0.1)  # Add a small delay to avoid excessive CPU usage
        except Exception as e:
            print(f"Exception in monitoring thread: {e}")
            time.sleep(1)  # Continue the loop even after an exception
def grasp():
    # while True:
    #     time.sleep(1)
    #     print("In grasp code")

    depth_img_que = Queue(1)
    ggcnn_cmd_que = Queue(1)
    try:
        camera = RealSenseCamera(width=WIDTH, height=HEIGHT)
        _, depth_intrin = camera.get_intrinsics()
        ggcnn = TorchGGCNN(depth_img_que, ggcnn_cmd_que, depth_intrin, width=WIDTH, height=HEIGHT, run_in_thread=GGCNN_IN_THREAD)
        fx = depth_intrin.fx
        fy = depth_intrin.fy
        cx = depth_intrin.ppx
        cy = depth_intrin.ppy
        time.sleep(3)
        grasp = RobotGrasp(robot_ip, ggcnn_cmd_que, EULER_EEF_TO_COLOR_OPT, EULER_COLOR_TO_DEPTH_OPT, GRASPING_RANGE, DETECT_XYZ, GRIPPER_Z_MM, RELEASE_XYZ, GRASPING_MIN_Z)

        color_image, depth_image = camera.get_images()
        color_shape = color_image.shape

        while grasp.is_alive():
            color_image, depth_image = camera.get_images()
            # cv2.imshow(WIN_NAME, color_image) # 显示彩色图像
            if GGCNN_IN_THREAD:
                if not depth_img_que.empty():
                    depth_img_que.get()
                depth_img_que.put([depth_image, grasp.CURR_POS[2] / 1000.0])
                cv2.imshow(WIN_NAME, color_image) # 显示彩色图像
            else:
                data = ggcnn.get_grasp_img(depth_image, cx, cy, fx, fy, grasp.CURR_POS[2] / 1000.0)
                if data:
                    if not ggcnn_cmd_que.empty():
                        ggcnn_cmd_que.get()
                    ggcnn_cmd_que.put(data[0])
                    grasp_img = data[1]
                    combined_img = np.zeros((color_shape[0], color_shape[1] + grasp_img.shape[1] + 10, 3), np.uint8)
                    combined_img[:color_shape[0], :color_shape[1]] = color_image
                    combined_img[:grasp_img.shape[0], color_shape[1]+10:color_shape[1]+grasp_img.shape[1]+10] = grasp_img

                    cv2.imshow(WIN_NAME, combined_img) # 显示彩色图像
                else:
                    cv2.imshow(WIN_NAME, color_image) # 显示彩色图像
            
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                camera.stop()
                break
    finally:
        # Ensure camera stops properly before exiting the function
        if 'camera' in locals():
            camera.stop()
            # Add a small delay to allow threads to clean up
            time.sleep(0.2)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)    
    global robot_ip

    if len(sys.argv) < 2:
        print('Usage: {} {{robot_ip}}'.format(sys.argv[0]))
        exit(1)

    robot_ip = sys.argv[1]

    # Create and start the background thread to monitor navigation status
    # Add stop event for thread coordination
    stop_event = Event()
    monitor_thread = Thread(target=background_monitor, args=(stop_event,), daemon=True)
    monitor_thread.start()
    
    # Keep the main thread alive
    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting program")
        stop_event.set()
        monitor_thread.join(timeout=2)  # Wait for thread to exit

if __name__ == '__main__':
    main()
