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


# XARM parameters
# # top down
# GRASPING_RANGE = [-50, 680, -450, 400]  # [x_min, x_max, y_min, y_max]
# RELEASE_XYZ = [400, 300, 100]
# GRASPING_MIN_Z = -400
# GRIPPER_Z_MM = -25  # mm - accounting for the shifted camera
# Camera calibration result
# EULER_EEF_TO_COLOR_OPT = [0.015, -0.1082, -0.118, 0, math.radians(20), math.pi/2]  # xyzrpy meters_rad - new tilted mount
# EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]

# # parallel picking
# ******************************Check Hori Pickup Tag***********************
# trail1
# GRASPING_RANGE = [-450, 850, -400, 400]  # [x_min, x_max, y_min, y_max]
# RELEASE_XYZ = [195, -420, 47]
# GRASPING_MIN_Z = 180
# trail 2
# GRASPING_RANGE = [-450, 1000, -300, 250]  # [x_min, x_max, y_min, y_max]
# RELEASE_XYZ = [195, -420, 47]
# GRASPING_MIN_Z = 480

# GRIPPER_Z_MM = 20  # mm - accounting for the shifted camera
# # Camera calibration result
# EULER_EEF_TO_COLOR_OPT = [-0.015, -0.1082, 0.118, 0, math.radians(-70),0]  # xyzrpy meters_rad - new tilted mount
# EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]

#trail 3
GRASPING_RANGE = [250, 850, -400, 400]  # [x_min, x_max, y_min, y_max]
RELEASE_XYZ = [195, -420, 47]
GRASPING_MIN_Z = 440

GRIPPER_Z_MM = 20  # mm - accounting for the shifted camera
# Camera calibration result
EULER_EEF_TO_COLOR_OPT = [-0.015, -0.1082, 0.118, 0, math.radians(-70),0]  # xyzrpy meters_rad - new tilted mount
EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]


DETECT_XYZ = [300, -200, 350]  # [x, y, z] # reset later in the code based on init pose
USE_INIT_POS = True
ARM_IP = '172.16.0.13'


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
                          GRASPING_RANGE, DETECT_XYZ, GRIPPER_Z_MM, RELEASE_XYZ, GRASPING_MIN_Z, hori_pickup=True)

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

                    # Extract grasp data parameters
                    x, y, z, ang, width, depth_center = data[0]
                    
                    # Convert the camera coordinates back to image coordinates
                    grasp_pixel_x = int(x * fx / z + cx)
                    grasp_pixel_y = int(y * fy / z + cy)
                    
                    # Draw grasp point on the color image portion
                    cv2.circle(combined_img, (grasp_pixel_x, grasp_pixel_y), 5, (0, 255, 0), -1)
                    
                    # Draw grasp orientation line
                    line_length = 25
                    end_x = int(grasp_pixel_x + line_length * np.cos(ang))
                    end_y = int(grasp_pixel_y + line_length * np.sin(ang))
                    cv2.line(combined_img, (grasp_pixel_x, grasp_pixel_y), (end_x, end_y), (0, 255, 0), 2)
                    
                    # Draw rectangle representing gripper width
                    rect_half_width = int(width / 2)
                    rect_height = 15  # arbitrary height for visualization
                    
                    # Calculate corner points of rectangle
                    # First create points for a horizontal rectangle
                    corners = [
                        [-rect_half_width, -rect_height//2],  # top-left
                        [rect_half_width, -rect_height//2],   # top-right
                        [rect_half_width, rect_height//2],    # bottom-right
                        [-rect_half_width, rect_height//2]    # bottom-left
                    ]
                    
                    # Rotate corners by ang
                    rotated_corners = []
                    cos_ang = np.cos(ang)
                    sin_ang = np.sin(ang)
                    
                    for cx, cy in corners:
                        # Rotate point around origin
                        rx = cx * cos_ang - cy * sin_ang
                        ry = cx * sin_ang + cy * cos_ang
                        
                        # Translate to center point
                        rx += grasp_pixel_x
                        ry += grasp_pixel_y
                        
                        # Ensure coordinates are within image bounds
                        rx = max(0, min(color_shape[1]-1, int(rx)))
                        ry = max(0, min(color_shape[0]-1, int(ry)))
                        
                        rotated_corners.append((rx, ry))
                    
                    # Draw lines between corners
                    for i in range(4):
                        start_point = rotated_corners[i]
                        end_point = rotated_corners[(i+1) % 4]
                        cv2.line(combined_img, start_point, end_point, (0, 255, 0), 2)
                    
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