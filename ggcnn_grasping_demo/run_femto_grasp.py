import sys
import cv2
import time
import numpy as np
import math
from queue import Queue

from camera.femto_camera import FemtboBoltCamera
from grasp.ggcnn_torch import TorchGGCNN
from grasp.robot_grasp import RobotGrasp

# Constants
WIN_NAME = 'FemtoBolt'
COLOR_DIM = [1280,960]
DEPTH_DIM = [1024,1024]
GGCNN_IN_THREAD = False

# Camera calibration result
EULER_EEF_TO_COLOR_OPT = [0.015, -0.1082, -0.118, 0, math.radians(20), math.pi/2]  # xyzrpy meters_rad - new tilted mount
EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0] # prealigned in camera code

# SARM parameters
GRASPING_RANGE = [-50, 680, -450, 400]  # [x_min, x_max, y_min, y_max]
GRIPPER_Z_MM = -25  # mm - accounting for the shifted camera
RELEASE_XYZ = [400, 300, 100]
GRASPING_MIN_Z = -400
DETECT_XYZ = [300, -200, 350]  # [x, y, z] # reset later in the code based on init pose
USE_INIT_POS = True

ARM_IP = '172.16.0.13'

# grasp_pose_list = []
def perform_grasp(arm_ip):
    """Main grasping function that operates the camera and arm"""
    depth_img_que = Queue(1)
    ggcnn_cmd_que = Queue(1)
    camera = None
    
    try:
        # Initialize camera
        camera = FemtboBoltCamera(color_wh=COLOR_DIM, depth_wh=DEPTH_DIM)
        _, depth_intrin = camera.get_intrinsics()
        print(depth_intrin)
        
        # Initialize GGCNN model
        ggcnn = TorchGGCNN(depth_img_que, ggcnn_cmd_que, depth_intrin, width=DEPTH_DIM[0], height=DEPTH_DIM[1], run_in_thread=GGCNN_IN_THREAD)
        fx = depth_intrin.fx
        fy = depth_intrin.fy
        cx = depth_intrin.cx
        cy = depth_intrin.cy
        
        # Wait for initialization
        time.sleep(3)
        print(fx,fy,cx,cy)
        # Initialize robot
        grasp = RobotGrasp(arm_ip, ggcnn_cmd_que, EULER_EEF_TO_COLOR_OPT, EULER_COLOR_TO_DEPTH_OPT, 
                          GRASPING_RANGE, DETECT_XYZ, GRIPPER_Z_MM, RELEASE_XYZ, GRASPING_MIN_Z)

        # Get initial images
        color_image, depth_image = camera.get_images()
        color_shape = color_image.shape

        # Main grasping loop
        while grasp.is_alive():
            color_image, depth_image = camera.get_images()
            # get the current eef position
            cur_eef_pos = []
            cur_eef_pos = grasp.get_eef_pose_m()
            print("cur_eef_pos: ", cur_eef_pos)
            cur_z = grasp.CURR_POS[2] / 1000.0
            
            # data = ggcnn.get_grasp_img(depth_image, cx, cy, fx, fy, cur_z)
            # pre_grasp = data[0]
            # print("pre_grasp: ", pre_grasp)
            if GGCNN_IN_THREAD:
                if not depth_img_que.empty():
                    depth_img_que.get()
                depth_img_que.put([depth_image, grasp.CURR_POS[2] / 1000.0])
                cv2.imshow(WIN_NAME, color_image)
            else:
                data = ggcnn.get_grasp_img(depth_image, cx, cy, fx, fy, cur_z)
                # post_grasp = data[0]
                # print("post and pre sleep grasp didn't changed: ", post_grasp == pre_grasp)
                time.sleep(6)
                if data:
                    cur_data = []
                    if not ggcnn_cmd_que.empty():
                        ggcnn_cmd_que.get()
                    cur_data = data[0]
                    cur_data.append(cur_eef_pos)
                    # print("cur_data: ",cur_data)
                    # print("data: ", cur_data[6])
                    # print("length of cur_data: ", len(cur_data))
                    ggcnn_cmd_que.put(cur_data)
                    # grasp_pose_list.append(data[0])
                    # time.sleep(60)
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
        # print(grasp_pose_list)
        if camera:
            camera.stop()
            # Add a small delay to allow threads to clean up
            time.sleep(0.2)

def run():
    """Run the grasping functionality without NetworkTables trigger"""
    if len(sys.argv) < 2:
        print(f"Assiging default IP: {ARM_IP}")
        arm_ip = ARM_IP
    else:
        arm_ip = sys.argv[1]

    perform_grasp(arm_ip)
    return 0

if __name__ == '__main__':
    sys.exit(run())