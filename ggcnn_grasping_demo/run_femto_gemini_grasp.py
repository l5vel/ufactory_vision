import sys
import cv2
import time
import numpy as np
import math
from queue import Queue
import threading
import PIL.Image as Image
import io
import matplotlib.pyplot as plt
# Import camera module
from camera.rs_camera import RealSenseCamera

# Import grasp modules
from grasp.ggcnn_torch import TorchGGCNN
from grasp.robot_grasp import RobotGrasp
from gemini_bbox import GeminiBbox
from gemma_bbox import Gemma3Bbox

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
# GRASPING_RANGE = [-450, 850, -400, 400]  # [x_min, x_max, y_min, y_max]
# RELEASE_XYZ = [195, -420, 47]
# GRASPING_MIN_Z = 180

GRASPING_RANGE = [250, 950, -400, 400]  # [x_min, x_max, y_min, y_max]
RELEASE_XYZ = [195, -420, 47]
GRASPING_MIN_Z = 440

GRIPPER_Z_MM = 35  # mm - accounting for the shifted camera
# Camera calibration result
EULER_EEF_TO_COLOR_OPT = [-0.12, 0.015, 
                          0.1082,    
                           0, math.radians(0),0]  # xyzrpy meters_rad - new tilted mount
# EULER_EEF_TO_COLOR_OPT = [-0.14, 0.015, 
#                         #   0.1082,
#                            0.06,          
#                            0, math.radians(0),0]  # xyzrpy meters_rad - flat mount
# EULER_EEF_TO_COLOR_OPT = [0,0,0,0,0,0]
EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]

DETECT_XYZ = [300, -200, 350]  # [x, y, z] # reset later in the code based on init pose
USE_INIT_POS = True
ARM_IP = '172.16.0.13'

HORI_PICKUP_BOOL = True
GGCNN_CUTOFF_DIST = 0.4 # distance from object at which the ggcnn will be stopped
ARMCTRL_MAX_DIST = 1.5 # max distance up to which arm control is triggered

class GeminiGGCNNGrasp:
    """
    Class that integrates all components: camera, GGCNN, robot control, and Gemini object detection
    """
    def __init__(self, arm_ip=ARM_IP, width=WIDTH, height=HEIGHT, use_thread=GGCNN_IN_THREAD):
        self.arm_ip = arm_ip
        self.width = width
        self.height = height
        self.use_thread = use_thread
        
        # Component objects
        self.camera = None
        self.ggcnn = None
        self.robot_grasp = None
        self.gemini_detector = None
        
        # Queues for thread-based operation
        self.depth_img_que = Queue(1)
        self.ggcnn_cmd_que = Queue(1)
        
        # Camera and robot parameters
        self.depth_intrin = None
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        
        # State variables
        self.running = False
        self.visualization_mode = "combined"  # "combined", "color", "grasp"
        self.run_ggcnn_gemini = True # tag to trigger grasp point calculation
        
    def initialize(self):
        """Initialize all system components"""
        try:
            # Initialize camera
            print("Initializing camera...")
            self.camera = RealSenseCamera(width=self.width, height=self.height)
            _, self.depth_intrin = self.camera.get_intrinsics()
            
            # Store intrinsic parameters for convenience
            self.fx = self.depth_intrin.fx
            self.fy = self.depth_intrin.fy
            self.cx = self.depth_intrin.ppx
            self.cy = self.depth_intrin.ppy
            
            print("fx, fy, cx, cy: ", self.fx, self.fy, self.cx, self.cy)
            # Wait for initialization
            time.sleep(3)
            
            # Initialize GGCNN model
            print("Initializing GGCNN model...")
            self.ggcnn = TorchGGCNN(
                depth_img_que=self.depth_img_que, 
                ggcnn_cmd_que=self.ggcnn_cmd_que, 
                depth_intrin=self.depth_intrin, 
                width=self.width, 
                height=self.height, 
                run_in_thread=self.use_thread
            )
            
            # Initialize robot 
            print(f"Initializing robot arm at {self.arm_ip}...")
            self.robot_grasp = RobotGrasp(
                self.arm_ip, 
                self.ggcnn_cmd_que, 
                EULER_EEF_TO_COLOR_OPT, 
                EULER_COLOR_TO_DEPTH_OPT,
                GRASPING_RANGE, 
                DETECT_XYZ, 
                GRIPPER_Z_MM, 
                RELEASE_XYZ, 
                GRASPING_MIN_Z,
                hori_pickup=HORI_PICKUP_BOOL,
                cutoff_dist=GGCNN_CUTOFF_DIST, 
                max_allowable_dist=ARMCTRL_MAX_DIST
            )
            
            # Initialize Gemini object detector
            print("Initializing Gemini object detector...")
            self.gemini_detector = GeminiBbox()
            # self.gemini_detector = Gemma3Bbox()
            
            # Wait for initialization to complete
            time.sleep(2)
            return True
            
        except Exception as e:
            print(f"Initialization error: {e}")
            self.cleanup()
            return False

    def detect_object_gemini(self, color_image, target_object="ball"):
        """
        Detect an object in the color image using Gemini
        
        Args:
            color_image: RGB image from camera
            target_object: Object to detect
            
        Returns:
            dict: Detection results or None if no detection
        """
        if not self.gemini_detector:
            return None
            
        # Create detection prompt
        highc_detection_prompt = f"""
        Analyze the image and identify any instances of '{target_object}'.
        From these, determine which '{target_object}' instance you are most confident about (the clearest, most certain detection).
        Output **only** the bounding box coordinates of that single most confident '{target_object}' in the format:
        {target_object} : (x_min, y_min, x_max, y_max)
        (Note: (x_min, y_min) is the top-left corner and (x_max, y_max) is the bottom-right corner of the bounding box.)
        If no '{target_object}' is detected in the image, clearly output "No {target_object} found."
        """
        img_pil = self.gemini_detector.numpy_to_pil(color_image)
        return self.gemini_detector.run_gemini_detect(img_pil, highc_detection_prompt, visualize=False)
        # return self.gemini_detector.run_gemma3_detect(img_pil, highc_detection_prompt, visualize=True)

    def isolate_object(self, depth_image, bbox, shape=(300, 300), visualize=False):
        # Extract depth ROI for the detected object
        x_min, y_min, x_max, y_max = bbox
        
        # First ensure the bounding box coordinates are valid
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(depth_image.shape[1], x_max), min(depth_image.shape[0], y_max)
        
        # Extract the ROI from depth image
        roi_depth = depth_image[y_min:y_max, x_min:x_max]
        # Calculate dimensions
        roi_height, roi_width = roi_depth.shape
        
        # Handle NaN values in the depth image
        depth_nan = np.isnan(roi_depth)
        roi_depth[depth_nan] = 0
        
        # Set max depth for background
        max_depth = roi_depth.max() * 5 if roi_depth.max() > 0 else 10000
        roi_depth[depth_nan] = max_depth
        
        # Create output depth image with specified shape - all background
        padded_depth = np.full(shape, max_depth)
        
        # Instead of centering, preserve original position by mapping coordinates
        # If the padded image is the same size as the original, use direct coordinates
        if shape == depth_image.shape:
            start_y, start_x = y_min, x_min
        else:
            # Scale the coordinates to fit in the padded image if sizes differ
            scale_y = shape[0] / depth_image.shape[0]
            scale_x = shape[1] / depth_image.shape[1]
            start_y = int(y_min * scale_y)
            start_x = int(x_min * scale_x)
        
        # Make sure coordinates are within bounds
        start_y = max(0, min(start_y, shape[0] - 1))
        start_x = max(0, min(start_x, shape[1] - 1))
        
        # Calculate end positions based on ROI size
        end_y = min(shape[0], start_y + roi_height)
        end_x = min(shape[1], start_x + roi_width)
        roi_y_end = min(roi_height, end_y - start_y)
        roi_x_end = min(roi_width, end_x - start_x)
        
        # print("start_y, start_x ", start_y, start_x)
        # print("end_y, end_x ", end_y, end_x)
        # print("ROI shape:", roi_depth.shape)
        # print("ROI slice dimensions:", roi_y_end, roi_x_end)
        # print("Padded area dimensions:", start_y, start_x, end_y, end_x)
        
        # # Place ROI in the padded image at the original position
        # print([start_y,start_y+roi_y_end, start_x,start_x+roi_x_end])

        padded_depth[start_y:start_y+roi_y_end, start_x:start_x+roi_x_end] = roi_depth[:roi_y_end, :roi_x_end]
        
        # Create visualization for debugging
        if visualize:
            # Normalize depth image for visualization (0-255)
            depth_viz = np.copy(depth_image)
            depth_viz = np.nan_to_num(depth_viz, nan=0)
            depth_viz = (depth_viz / (depth_viz.max() or 1) * 255).astype(np.uint8)
            
            # Create mask for ROI area in original image
            mask = np.zeros_like(depth_image, dtype=np.uint8)
            mask[y_min:y_max, x_min:x_max] = 1
            
            # Draw rectangle on original visualization
            viz_with_rect = depth_viz.copy()
            cv2.rectangle(viz_with_rect, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Also visualize the padded output - using the original normalization
            padded_viz = (padded_depth / (padded_depth.max() or 1) * 255).astype(np.uint8)
            cv2.imshow("Isolated Object", padded_viz)
            cv2.waitKey(1)
        
        return padded_depth, [start_y, start_x, end_y, end_x]

    def get_grasp_on_object(self, color_image, depth_image, robot_z, target_object="object"):
        """
        Perform a complete grasp operation:
        1. Detect object using Gemini
        2. Calculate grasp points using GGCNN
        
        Args:
            target_object: Object to detect and grasp
            
        Returns:
            bool: True if grasp was successful, False otherwise
        """
        if not self.camera or not self.ggcnn or not self.robot_grasp:
            print("System not fully initialized")
            return [None, None], None, False
            
        try:
            # Detect object if the arm is ready to grasp
            if self.robot_grasp.init_pose is None:
                print("Robot arm initial pose not setup, skipping detect object")
                return [None, None], None, False
            
            detection = self.detect_object_gemini(color_image, target_object)
            if not detection:
                print(f"No {target_object} detected")
                return [None, None], None, False

            # print("detection result: ", detection)
            # Extract bounding box
            bbox = detection[0]['bounding_box']
            roi_depth, bbox_coords = self.isolate_object(depth_image, bbox, depth_image.shape, visualize=False) 
            # print("bbox_coords ", bbox_coords)
            # Process the depth image directly to get grasp data
            grasp_data, grasp_img = self.ggcnn.get_grasp_img(
                roi_depth, 
                self.cx, 
                self.cy, 
                self.fx, 
                self.fy, 
                robot_z,
                custom_crop=bbox_coords,
                visualize_crop=False
            )
            if not grasp_data:
                print("No valid grasp points found")
                return [None, None], None, False
                
            return [grasp_data, grasp_img], bbox_coords, True
            
        except Exception as e:
            print(f"Error in grasp operation: {e}")
            return [None, None], None, False
        
    def calc_depth_center(self, depth_image, roi_bbox=None):
        """
        Calculate the depth center of the given depth image
        
        Args:
            depth_image: Input depth image
            roi_bbox: Optional region of interest [start_y, start_x, end_y, end_x]
        
        Returns:
            depth_center: Average depth of the center region in mm
        """
        H, W = depth_image.shape
        
        if roi_bbox is not None:
            # Use the custom ROI if provided
            start_y, start_x, end_y, end_x = roi_bbox
            
            # Ensure bbox coordinates are within image bounds
            start_y = max(0, start_y)
            start_x = max(0, start_x)
            end_y = min(H, end_y)
            end_x = min(W, end_x)
            
            # Extract the ROI
            crop = depth_image[start_y:end_y, start_x:end_x]
            
            # Resize to 300x300 for consistent processing
            crop = cv2.resize(crop, (300, 300))
        else:
            # Use the default center crop method
            cs = 400  # Default crop size
            hs, ws = max(0, (H - cs)//2), max(0, (W - cs)//2)
            crop = cv2.resize(depth_image[hs:hs+cs, ws:ws+cs], (300, 300))
        
        # 2) Replace NaNs with 0 in one go
        crop = np.nan_to_num(crop, nan=0.0)
        
        # 3) Inpaint (pad→mask→scale→inpaint→unpad)
        pad = 1
        crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_DEFAULT)
        mask = (crop == 0).astype(np.uint8)
        scale = crop.max() if crop.max() > 0 else 1.0  # Prevent division by zero
        crop = (crop.astype(np.float32) / scale)
        crop = cv2.inpaint(crop, mask, 1, cv2.INPAINT_NS)[pad:-pad, pad:-pad] * scale
        
        # 4) Extract the 41×41 center patch, take the 10 smallest pixels, average and convert to mm
        vals = np.sort(crop[100:141, 130:171].ravel())[:10]
        
        # Handle case where there might not be enough valid depth values
        if len(vals) > 0:
            depth_center = vals.mean() * 1000.0
        else:
            depth_center = 0.0  # Default value if no valid depths found
        
        return depth_center
    
    def run(self, target_object = None):
        """Run the system in continuous mode, showing camera feed and grasp visualization"""
        if not self.initialize():
            return False
        prev_comb_img = None # save the previous combined image
        self.running = True
        combined_img = None
        try:
            # Main processing loop
            while self.running:
                if not self.robot_grasp.alive: # shutdown if robot is not alive
                   print("Robot arm not alive, shutting down...")
                   self.running = False
                   self.cleanup()
                   break
                # Get images from camera
                color_image, depth_image = self.camera.get_images()
                color_shape = color_image.shape
                
                if not HORI_PICKUP_BOOL:
                    # Get robot Z position if available
                    robot_z = self.robot_grasp.CURR_POS[2] / 1000.0 if self.robot_grasp else 0.5
                else:
                    # get the current eef position
                    cur_eef_pos = []
                    cur_eef_pos = self.robot_grasp.get_eef_pose_m()
                    # print("cur_eef_pos: ", cur_eef_pos)
                    robot_z = self.robot_grasp.CURR_POS[2] / 1000.0 if self.robot_grasp else 0.5
                # print("robot_z: ", robot_z)
                # Process frames based on thread mode
                if self.use_thread:
                    # Thread mode: use queues
                    if not self.depth_img_que.empty():
                        self.depth_img_que.get()
                    self.depth_img_que.put([depth_image, robot_z])
                    
                    # Show color image only
                    cv2.imshow(WIN_NAME, color_image)
                else:
                    roi_bbox = None
                    if target_object is not None:
                        # CHECK dist valid code to see if the distance is within limits is being reset
                        if self.robot_grasp.dist_valid: # this is false after range beyond cutoff
                            self.run_ggcnn_gemini = True  
                        else:
                            self.run_ggcnn_gemini = False

                        # print("GEmini running: ", self.run_ggcnn_gemini)
                        if self.run_ggcnn_gemini:
                            results, roi_bbox, grasp_bool = self.get_grasp_on_object(color_image, depth_image, robot_z, target_object)
                        else:
                            grasp_bool = False
                    else:
                        print("No target_object input received")
                        break
            
                    if grasp_bool:
                        grasp_data, grasp_img = results
                        # adjust the depth center to correspond to the appropriate depth image
                        if roi_bbox is not None:
                            grasp_data[5] = self.calc_depth_center(depth_image, roi_bbox = roi_bbox)
                        else:
                            grasp_data[5] = self.calc_depth_center(depth_image)
                
                        if any(grasp_data) >= 1e4: # means the masking is bad
                            print("Bad grasp data, skipping...", grasp_data)
                            continue
                        # print("grasp_data: ",grasp_data)
                        # Add grasp data to queue for robot 
                        if self.robot_grasp:
                            if not self.ggcnn_cmd_que.empty():
                                self.ggcnn_cmd_que.get()
                            # self.ggcnn_cmd_que.put(grasp_data)
                            if not HORI_PICKUP_BOOL:
                                self.ggcnn_cmd_que.put(grasp_data)
                            else:
                                cur_data = []
                                cur_data = grasp_data
                                cur_data.append(cur_eef_pos)
                                # print("cur_data: ",cur_data)
                                # print("data: ", cur_data[6])
                                self.ggcnn_cmd_que.put(cur_data)
                                # print("length of cur_data:************** ", len(cur_data))
                                
                        # Display different visualizations based on mode
                        if self.visualization_mode == "combined":
                            # Create the combined image
                            combined_img = np.zeros((color_shape[0], color_shape[1] + grasp_img.shape[1] + 10, 3), np.uint8)
                            combined_img[:color_shape[0], :color_shape[1]] = color_image
                            combined_img[:grasp_img.shape[0], color_shape[1]+10:color_shape[1]+grasp_img.shape[1]+10] = grasp_img
                            
                            # Extract grasp data parameters
                            if len(grasp_data) <= 6:
                                x, y, z, ang, width, depth_center = grasp_data
                            else:
                                x, y, z, ang, width, depth_center,_ = grasp_data
                            
                            grasp_pixel_x = int(x * self.fx / z + self.cx)
                            grasp_pixel_y = int(y * self.fy / z + self.cy)
                            
                            # Draw grasp point on the color image portion
                            cv2.circle(combined_img, (grasp_pixel_x, grasp_pixel_y), 5, (0, 255, 0), -1)
                            
                            cv2.circle(combined_img, (int(self.cx), int(self.cy)), 5, (0, 0, 255), -1)

                            cv2.circle(combined_img, (grasp_pixel_x, int(self.cy)), 5, (255, 0, 0), -1)
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
                            
                            # # Add text with grasp info
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            # cv2.putText(combined_img, f"Grasp: ({x:.2f}, {y:.2f}, {z:.2f})", 
                            #             (grasp_pixel_x+10, grasp_pixel_y-20), 
                            #             font, 0.5, (0, 255, 0), 1)
                            # cv2.putText(combined_img, f"Angle: {ang:.2f}, Width: {width:.2f}", 
                            #             (grasp_pixel_x+10, grasp_pixel_y-5), 
                            #             font, 0.5, (0, 255, 0), 1)
                            # print("roi_bbox ", roi_bbox)
                            if roi_bbox is not None:
                                # print("roi_bbox ", roi_bbox)
                                # Draw rectangle with cv2
                                cv2.rectangle(combined_img, (roi_bbox[1], roi_bbox[0]), (roi_bbox[3], roi_bbox[2]), (0, 255, 0), 3)  # (0, 255, 0) is lime color in BGR

                                cv2.putText(combined_img, f"{target_object}(d): {(grasp_data[5]/1000):.2f}m", 
                                        (roi_bbox[1]+10, roi_bbox[0]-5), 
                                        font, 0.5, (0, 255, 0), 1)
                            
                            prev_comb_img = combined_img

                            # Display the combined image
                            cv2.imshow(WIN_NAME, combined_img)
                        elif self.visualization_mode == "grasp":
                            cv2.imshow(WIN_NAME, grasp_img)
                        else:
                            cv2.imshow(WIN_NAME, color_image)
                    else:
                        # No valid grasp points - show last image if available
                        if combined_img is not None:
                            cv2.imshow(WIN_NAME, combined_img)
                        else:
                            cv2.imshow(WIN_NAME, color_image)

                
                # Process keyboard input
                key = cv2.waitKey(1)
                if key & 0xFF == ord('v'):  # 'v' to cycle visualization modes
                    modes = ["combined", "color", "grasp"]
                    current_index = modes.index(self.visualization_mode)
                    self.visualization_mode = modes[(current_index + 1) % len(modes)]
                    
            return True
            
        except Exception as e:
            print(f"Runtime error: {e}")
            return False
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        # Stop camera
        if self.camera:
            self.camera.stop()
            
        # Close windows
        cv2.destroyAllWindows()
        
        # Wait for threads to clean up
        time.sleep(0.2)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {} {{arm_ip}} {{target_object}}'.format(sys.argv[0]))
        sys.exit(0)
        
    arm_ip = sys.argv[1]
    
    # Parse optional object parameter
    target_object = "object"
    if len(sys.argv) >= 3:
        target_object = sys.argv[2]
    
    # Create grasp system
    grasp_system = GeminiGGCNNGrasp(arm_ip=arm_ip)
    
    grasp_system.run(target_object=target_object)
    sys.exit(0)