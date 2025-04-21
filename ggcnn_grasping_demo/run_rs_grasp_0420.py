import sys
import cv2
import time
import numpy as np
import math
from queue import Queue

from camera.rs_camera import RealSenseCamera
from grasp.ggcnn_torch import TorchGGCNN
from grasp.robot_grasp import RobotGrasp

# Constants
WIN_NAME = 'RealSense'
WIDTH = 640
HEIGHT = 480
GGCNN_IN_THREAD = False

# Camera calibration result
EULER_EEF_TO_COLOR_OPT = [0.015, -0.1082, -0.118, 0, math.radians(20), math.pi/2]  # xyzrpy meters_rad - new tilted mount
EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]

# SARM parameters
GRASPING_RANGE = [-50, 680, -450, 400]  # [x_min, x_max, y_min, y_max]
GRIPPER_Z_MM = -25  # mm - accounting for the shifted camera
RELEASE_XYZ = [400, 300, 100]
GRASPING_MIN_Z = -400
DETECT_XYZ = [300, -200, 350]  # [x, y, z] # reset later in the code based on init pose
USE_INIT_POS = True

def perform_grasp(arm_ip):
    """Main grasping function that operates the camera and arm"""
    depth_img_que = Queue(1)
    ggcnn_cmd_que = Queue(1)
    camera = None
    
    try:
        # Initialize camera
        camera = RealSenseCamera(width=WIDTH, height=HEIGHT)
        _, depth_intrin = camera.get_intrinsics()
        
        # Initialize GGCNN model
        ggcnn = TorchGGCNN(depth_img_que, ggcnn_cmd_que, depth_intrin, width=WIDTH, height=HEIGHT, run_in_thread=GGCNN_IN_THREAD)
        fx = depth_intrin.fx
        fy = depth_intrin.fy
        cx = depth_intrin.ppx
        cy = depth_intrin.ppy
        
        # Wait for initialization
        time.sleep(3)
        
        # Initialize robot
        grasp = RobotGrasp(arm_ip, ggcnn_cmd_que, EULER_EEF_TO_COLOR_OPT, EULER_COLOR_TO_DEPTH_OPT, 
                          GRASPING_RANGE, DETECT_XYZ, GRIPPER_Z_MM, RELEASE_XYZ, GRASPING_MIN_Z)

        # Get initial images
        color_image, depth_image = camera.get_images()
        color_shape = color_image.shape

        # Main grasping loop
        while grasp.is_alive():
            color_image, depth_image = camera.get_images()
            
            if GGCNN_IN_THREAD:
                if not depth_img_que.empty():
                    depth_img_que.get()
                depth_img_que.put([depth_image, grasp.CURR_POS[2] / 1000.0])
                cv2.imshow(WIN_NAME, color_image)
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

                    cv2.imshow(WIN_NAME, combined_img)
                else:
                    cv2.imshow(WIN_NAME, color_image)
            
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                break
                
        return True  # Indicate successful completion
        
    except Exception as e:
        print(f"Exception in grasping operation: {e}")
        return False
        
    finally:
        # Ensure camera stops properly before exiting the function
        if camera:
            camera.stop()
            # Add a small delay to allow threads to clean up
            time.sleep(0.2)

def run():
    """Run the grasping functionality without NetworkTables trigger"""
    if len(sys.argv) < 2:
        print('Usage: {} {{arm_ip}}'.format(sys.argv[0]))
        return 1

    arm_ip = sys.argv[1]
    perform_grasp(arm_ip)
    return 0

if __name__ == '__main__':
    sys.exit(run())