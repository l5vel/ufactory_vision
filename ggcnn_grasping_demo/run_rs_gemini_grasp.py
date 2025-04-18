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

# Constants
WIN_NAME = 'RealSense'
WIDTH = 640
HEIGHT = 480
GGCNN_IN_THREAD = False

# Camera calibration result
EULER_EEF_TO_COLOR_OPT = [0.015, -0.1082, -0.118, 0, math.radians(20), math.pi/2]  # xyzrpy meters_rad - tilted mount
EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]

# Robot arm parameters
GRASPING_RANGE = [-50, 680, -450, 400]  # [x_min, x_max, y_min, y_max]
GRIPPER_Z_MM = -25  # mm - accounting for the shifted camera
RELEASE_XYZ = [400, 350, 400]
GRASPING_MIN_Z = -400
DETECT_XYZ = [300, -200, 350]  # [x, y, z] # reset later based on init pose
USE_INIT_POS = True
ARM_IP = '172.16.0.13'

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
                GRASPING_MIN_Z
            )
            
            # Initialize Gemini object detector
            print("Initializing Gemini object detector...")
            self.gemini_detector = GeminiBbox()
            
            # Wait for initialization to complete
            time.sleep(2)
            return True
            
        except Exception as e:
            print(f"Initialization error: {e}")
            self.cleanup()
            return False
    
    def numpy_to_pil(self,img_np):
        """Converts a NumPy array to a PIL Image."""
        if img_np.ndim == 2:  # Handle grayscale images
            img_pil = Image.fromarray(img_np, 'L')
        elif img_np.ndim == 3:  # Handle color images
            img_pil = Image.fromarray(img_np, 'RGB') # Or 'BGR' depending on OpenCV's default
        else:
            raise ValueError(f"Unsupported image ndim: {img_np.ndim}")
        return img_pil

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
        img_pil = self.numpy_to_pil(color_image)
        return self.gemini_detector.run_gemini_detect(img_pil, highc_detection_prompt, visualize=False)

    def isolate_object(self, depth_image, bbox, shape=(300, 300)):
        # Extract depth ROI for the detected object
        x_min, y_min, x_max, y_max = bbox
        # print(f"Bounding box: {bbox}")
        # First ensure the bounding box coordinates are valid
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(depth_image.shape[1], x_max), min(depth_image.shape[0], y_max)

        # use bbox to crop the depth image and pad it to 640x480
        roi_depth = depth_image[y_min:y_max, x_min:x_max]
        # Handle NaN values in the depth image
        depth_nan = np.isnan(roi_depth)
        # max excluding NaN values
        roi_depth[depth_nan] = 0
        max_depth = roi_depth.max()*5
        # replace NaN values with max_dept
        # roi_depth[depth_nan] = max_depth
        roi_depth[depth_nan] = max_depth
        
        # make everything else in the 640x480 image - max_depth
        padded_depth = np.full(shape, max_depth)
        padded_depth[y_min:y_min+roi_depth.shape[0], x_min:x_min+roi_depth.shape[1]] = roi_depth
        
        # Create a combined visualization
        # 1. Convert depth image to visualization (normalized to 0-255)
        depth_viz = np.copy(depth_image)


        # 3. Make the ROI stand out
        roi_viz = np.copy(padded_depth)
        roi_viz = (roi_viz / np.max(roi_viz) * 255).astype(np.uint8)
        
        # Create mask for ROI area
        mask = np.zeros_like(depth_image, dtype=np.uint8)
        mask[y_min:y_min+roi_depth.shape[0], x_min:x_min+roi_depth.shape[1]] = 1
         
        # 4. Combine images
        combined_viz = np.where((mask > 0) & (roi_viz < max_depth), roi_viz, depth_viz)
        
        # 5. Add rectangle around ROI on combined visualization
        cv2.rectangle(combined_viz, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        

        cv2.imshow("depth_image", combined_viz)
        cv2.waitKey(1)
        return padded_depth

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
            return [None, None], False
            
        try:
            # Detect object
            detection = self.detect_object_gemini(color_image, target_object)
            if not detection:
                print(f"No {target_object} detected")
                return [None, None], False

            # print("detection result: ", detection)
            # Extract bounding box
            bbox = detection[0]['bounding_box']
            roi_depth = self.isolate_object(depth_image, bbox, depth_image.shape) 
            # Process the depth image directly to get grasp data
            grasp_data, grasp_img = self.ggcnn.get_grasp_img(
                roi_depth, 
                self.cx, 
                self.cy, 
                self.fx, 
                self.fy, 
                robot_z
            )
            if not grasp_data:
                print("No valid grasp points found")
                return [None, None], False
                
            return [grasp_data, grasp_img], True
            
        except Exception as e:
            print(f"Error in grasp operation: {e}")
            return [None, None], False
        
    def calc_depth_center(self, depth_image):
        # based on the code in ggccnn_torch.py
        H, W = depth_image.shape
        cs = 400
        hs, ws = max(0, (H - cs)//2), max(0, (W - cs)//2)
        crop = cv2.resize(depth_image[hs:hs+cs, ws:ws+cs], (300, 300))

        # 2) Replace NaNs with 0 in one go
        crop = np.nan_to_num(crop, nan=0.0)

        # 3) Inpaint (pad→mask→scale→inpaint→unpad)
        pad = 1
        crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_DEFAULT)
        mask = (crop == 0).astype(np.uint8)
        scale = crop.max()
        crop = (crop.astype(np.float32) / scale)
        crop = cv2.inpaint(crop, mask, 1, cv2.INPAINT_NS)[pad:-pad, pad:-pad] * scale

        # 4) Extract the 41×41 center patch, take the 10 smallest pixels, average and convert to mm
        vals = np.sort(crop[100:141, 130:171].ravel())[:10]
        depth_center = vals.mean() * 1000.0
        return depth_center
    
    def run(self, target_object = None):
        """Run the system in continuous mode, showing camera feed and grasp visualization"""
        if not self.initialize():
            return False
            
        self.running = True
        
        try:
            # Main processing loop
            while self.running:
                # Get images from camera
                color_image, depth_image = self.camera.get_images()
                color_shape = color_image.shape
                
                # Get robot Z position if available
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
                    if target_object is not None:
                        results, grasp_bool = self.get_grasp_on_object(color_image, depth_image, robot_z, target_object)
                    else:
                        print("No target_object input received")
                        break
                    
                    if grasp_bool:
                        grasp_data, grasp_img = results
                        
                        # adjust the depth center to correspond to the original depth image
                        grasp_data[5] = self.calc_depth_center(depth_image)
                
                        if any(grasp_data) >= 1e4: # means the masking is bad
                            print("Bad grasp data, skipping...", grasp_data)
                            continue
                        # print("grasp_data: ",grasp_data)
                        # Add grasp data to queue for robot 
                        if self.robot_grasp:
                            if not self.ggcnn_cmd_que.empty():
                                self.ggcnn_cmd_que.get()
                            self.ggcnn_cmd_que.put(grasp_data)
                        
                        # Display different visualizations based on mode
                        if self.visualization_mode == "combined":
                            combined_img = np.zeros((color_shape[0], color_shape[1] + grasp_img.shape[1] + 10, 3), np.uint8)
                            combined_img[:color_shape[0], :color_shape[1]] = color_image
                            combined_img[:grasp_img.shape[0], color_shape[1]+10:color_shape[1]+grasp_img.shape[1]+10] = grasp_img
                            cv2.imshow(WIN_NAME, combined_img)
                        elif self.visualization_mode == "grasp":
                            cv2.imshow(WIN_NAME, grasp_img)
                        else:
                            cv2.imshow(WIN_NAME, color_image)
                    else:
                        # No valid grasp points
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