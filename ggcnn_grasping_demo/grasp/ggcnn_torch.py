import sys
import time
import cv2
import torch
import numpy as np
import scipy.ndimage as ndimage
from skimage.draw import disk, line
from skimage.feature import peak_local_max
import threading

sys.path.append('./ggcnn')


# Execution Timing
class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = False

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        self.t1 = time.time()
        if self.print_output:
            print('%s: %s' % (self.s, self.t1 - self.t0))


class TorchGGCNN(object):
    def __init__(self, depth_img_que=None, ggcnn_cmd_que=None, depth_intrin=None, width=640, height=480, run_in_thread=False):
        self.device = torch.device('cuda')
        # self.model = torch.load('/home/l5vel/ufactory_vision/cornelltune_03052', map_location=self.device, weights_only=False)
        self.model = torch.load('/home/l5vel/ufactory_vision/cornelltune_06053', map_location=self.device, weights_only=False)
        
        # self.model = torch.load('/home/l5vel/ufactory_vision/ggcnn_grasping_demo/models/ggcnn_epoch_23_cornell', map_location=self.device, weights_only=False)
        self.robot_z = 0.5
        self.prev_mp = np.array([150, 150])
        self.depth_img_que = depth_img_que
        self.ggcnn_cmd_que = ggcnn_cmd_que
        self.depth_intrin = depth_intrin
        self.width = width
        self.height = height
        if run_in_thread and depth_img_que is not None and ggcnn_cmd_que is not None:
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def draw_rectangle(self, x, y, ang, width, grasp_img):
        """
        Draw a rectangle at specified coordinates with given angle and width
        
        Args:
            x, y: Center coordinates of the rectangle
            ang: Angle of the rectangle in radians
            width: Width of the rectangle
            grasp_img: Image to draw on
        """
        # Define rectangle parameters
        rect_height = 15 # arbritrary
        rect_color = (0, 255, 0)  # Green
        
        # Calculate corner points of rectangle
        # First create points for a horizontal rectangle
        half_width = width // 2
        half_height = rect_height // 2
        
        # Corner points relative to center (before rotation)
        corners = [
            [-half_width, -half_height],  # top-left
            [half_width, -half_height],   # top-right
            [half_width, half_height],    # bottom-right
            [-half_width, half_height]    # bottom-left
        ]
        
        # Rotate corners by ang
        rotated_corners = []
        cos_ang = np.cos(ang)
        sin_ang = np.sin(ang)
        
        for cx, cy in corners:
            # Rotate point around origin
            rx = cx * cos_ang - cy * sin_ang
            ry = cx * sin_ang + cy * cos_ang
            
            # Translate to center point (x,y)
            rx += x
            ry += y
            
            # Ensure coordinates are within image bounds
            rx = max(0, min(grasp_img.shape[1]-1, int(rx)))
            ry = max(0, min(grasp_img.shape[0]-1, int(ry)))
            
            rotated_corners.append((rx, ry))
        
        # Draw lines between corners
        for i in range(4):
            start_point = rotated_corners[i]
            end_point = rotated_corners[(i+1) % 4]
            
            # Use Bresenham's line algorithm
            rr, cc = line(start_point[1], start_point[0], 
                        end_point[1], end_point[0])
            
            # Make sure all coordinates are within image bounds
            valid_indices = (
                (rr >= 0) & (rr < grasp_img.shape[0]) & 
                (cc >= 0) & (cc < grasp_img.shape[1])
            )
            rr, cc = rr[valid_indices], cc[valid_indices]
            
            # Set pixel values
            grasp_img[rr, cc, 0] = rect_color[0]
            grasp_img[rr, cc, 1] = rect_color[1]
            grasp_img[rr, cc, 2] = rect_color[2]
            
        return grasp_img

    def run(self):
        fx = self.depth_intrin.fx
        fy = self.depth_intrin.fy
        cx = self.depth_intrin.ppx
        cy = self.depth_intrin.ppy
        while True:
            data = self.depth_img_que.get()
            img, robot_z = data[0], data[1]
            data = self.get_grasp_img(img, cx, cy, fx, fy, robot_z)
            if data:
                if not self.ggcnn_cmd_que.empty():
                    self.ggcnn_cmd_que.get()
                self.ggcnn_cmd_que.put(data[0])

    def get_grasp_img(self, depth_image, cx, cy, fx, fy, robot_z=None, custom_crop=None, target_size=300, visualize_crop=False):
        """
        Process depth image to find optimal grasp points
        
        Args:
            depth_image: Input depth image
            cx, cy, fx, fy: Camera intrinsic parameters
            robot_z: Robot Z position (optional)
            custom_crop: Custom crop coordinates [y_min, x_min, y_max, x_max] (optional)
            target_size: Target size for processing (default 300)
            visualize_crop: Whether to visualize the crop area on the original image
        
        Returns:
            Grasp parameters, visualization image, and (optional) crop visualization
        """
        if robot_z is not None:
            self.robot_z = robot_z
            
        with TimeIt('Crop'):
            # Set default crop based on input size if no custom crop is provided
            if custom_crop is None:
                # Default crop settings - center crop
                crop_size = min(400, min(self.height, self.width))  # Default to 400 or smaller if image is smaller
                height_crop = max(0, self.height - crop_size)
                width_crop = max(0, self.width - crop_size)
                
                y_min = height_crop//2
                x_min = width_crop//2
                y_max = y_min + crop_size
                x_max = x_min + crop_size
            else:
                # Use custom crop if provided
                y_min, x_min, y_max, x_max = custom_crop
                # Ensure crop is within image bounds
                y_min = max(0, y_min)
                x_min = max(0, x_min)
                y_max = min(self.height, y_max)
                x_max = min(self.width, x_max)
            
            # Extract the ROI directly from the specified coordinates
            depth_crop = depth_image[y_min:y_max, x_min:x_max]
            # Store crop coordinates for later reference when converting back
            crop_origin = np.array([y_min, x_min])
            actual_crop_height = y_max - y_min
            actual_crop_width = x_max - x_min
            
            # Store original crop dimensions for coordinate conversion later
            original_crop_dims = (actual_crop_height, actual_crop_width)
            
            # Only resize if needed (if crop size is not already target_size x target_size)
            if actual_crop_height != target_size or actual_crop_width != target_size:
                depth_crop = cv2.resize(depth_crop, (target_size, target_size))
                
            # # Print debug info
            # print("Crop origin (y, x):", crop_origin)
            # print("Actual crop dimensions (h, w):", actual_crop_height, actual_crop_width)
            # print("Crop rectangle coordinates [y_min, x_min, y_max, x_max]:", y_min, x_min, y_max, x_max)
            # print("Target processing size:", target_size)
                
            # Replace nan with 0 for inpainting
            depth_crop = depth_crop.copy()
            depth_nan = np.isnan(depth_crop).copy()
            depth_crop[depth_nan] = 0
            
            # Create visualization showing the crop area on the original image
            if visualize_crop:
                # Create a colored version of the depth image for visualization
                depth_viz = np.copy(depth_image)
                depth_viz = np.nan_to_num(depth_viz, nan=0)
                depth_viz = (depth_viz / (depth_viz.max() or 1) * 255).astype(np.uint8)
                
                # Convert grayscale to RGB for colored box drawing
                depth_viz_color = cv2.cvtColor(depth_viz, cv2.COLOR_GRAY2BGR)
                
                # Draw a rectangle on the original image to show the crop area
                cv2.rectangle(depth_viz_color, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                
                # Add text labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(depth_viz_color, f"Crop Area", (x_min, y_min-10), font, 0.5, (0, 0, 255), 1)
                cv2.putText(depth_viz_color, f"({x_min},{y_min})", (x_min, y_max+15), font, 0.5, (0, 0, 255), 1)
                cv2.putText(depth_viz_color, f"({x_max},{y_max})", (x_max-70, y_max+15), font, 0.5, (0, 0, 255), 1)
                
                # # Display the visualization
                # cv2.imshow("Original With Crop", depth_viz_color)
                # cv2.waitKey(1)

        # Rest of the function remains the same until the Control section
        with TimeIt('Inpaint'):
            # open cv inpainting does weird things at the border.
            depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
            mask = (depth_crop == 0).astype(np.uint8)
            # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
            depth_scale = np.abs(depth_crop).max()
            depth_crop = depth_crop.astype(np.float32)/depth_scale  # Has to be float32, 64 not supported.
        
            depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS)
            # Back to original size and value range.
            depth_crop = depth_crop[1:-1, 1:-1]
            depth_crop = depth_crop * depth_scale
        
        with TimeIt('Calculate Depth'): 
            # Calculate center region for depth estimation - adjust if target_size is not 300
            center_ratio = 0.33  # Use ~1/3 of the image center for depth estimation
            center_size = int(target_size * center_ratio)
            center_start = (target_size - center_size) // 2
            center_end = center_start + center_size
            
            # Figure out roughly the depth in mm of the part between the grippers for collision avoidance.
            depth_center = depth_crop[center_start:center_end, center_start:center_end].flatten()
            depth_center.sort()
            depth_center = depth_center[:10].mean() * 1000.0
        
        with TimeIt('Inference'):
            # Run it through the network. Normalize:
            depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
            input_numpy_array = depth_crop.reshape((1, target_size, target_size, 1))
            input_tensor = torch.as_tensor(input_numpy_array).to(self.device)
            input_tensor = input_tensor.permute(0, 3, 1, 2)
            with torch.no_grad():
                pred_out = self.model(input_tensor)

            points_out = pred_out[0].cpu().numpy().squeeze()
            points_out[depth_nan] = 0
        
        with TimeIt('Trig'):
            # Calculate the angle map.
            cos_out = pred_out[1].cpu().numpy().squeeze()
            sin_out = pred_out[2].cpu().numpy().squeeze()
            ang_out = np.arctan2(sin_out, cos_out)/2.0

            width_out = pred_out[3].cpu().numpy().squeeze() * 150.0  # Scaled 0-150:0-1

        with TimeIt('Filter'):
            # Filter the outputs.
            points_out = ndimage.gaussian_filter(points_out, 5.0)
            ang_out = ndimage.gaussian_filter(ang_out, 2.0)
        
        with TimeIt('Control'):
            # Calculate the best pose from the camera intrinsics.
            maxes = None

            ALWAYS_MAX = False

            if self.robot_z > 0.34 or ALWAYS_MAX:
                # Track the global max.
                max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
                self.prev_mp = max_pixel.astype(np.int64)
            else:
                # Calculate a set of local maxes. Choose the one closest to the previous one.
                maxes = peak_local_max(points_out, min_distance=10, threshold_abs=0.1, num_peaks=3)
                if maxes.shape[0] == 0:
                    return
                max_pixel = maxes[np.argmin(np.linalg.norm(maxes - self.prev_mp, axis=1))]

                # Keep a global copy for next iteration.
                self.prev_mp = (max_pixel * 0.25 + self.prev_mp * 0.75).astype(np.int64)

            # Store max pixel in target_size x target_size coordinates for visualization
            max_pix_target = max_pixel
            
            # Get angle and width at the maximum point
            ang = ang_out[max_pixel[0], max_pixel[1]]
            width = width_out[max_pixel[0], max_pixel[1]]
            
            # DEBUG
            # print("Max pixel in target size:", max_pix_target)
            # print("Angle at max point:", ang)
            # print("Width at max point:", width)
            
            # Convert max_pixel back to original image coordinates
            # Scale factor from target_size to actual crop size
            scale_y = original_crop_dims[0] / float(target_size)
            scale_x = original_crop_dims[1] / float(target_size)
            
            # Apply scaling and add crop origin offset
            max_pixel_original = (np.array(max_pixel) * np.array([scale_y, scale_x])) + crop_origin
            max_pixel_original = np.round(max_pixel_original).astype(np.int64)
            
            # DEBUG
            # print("Max pixel in original coords:", max_pixel_original)
            
            # Ensure coordinates are within image bounds
            max_pixel_original[0] = max(0, min(self.height-1, max_pixel_original[0]))
            max_pixel_original[1] = max(0, min(self.width-1, max_pixel_original[1]))
            # print("self.width, self.height: ", self.width, self.height)
            # Get depth at this point
            point_depth = depth_image[max_pixel_original[0], max_pixel_original[1]]
            # print("*************Grasp Center Coordinates*************", max_pixel_original[0], max_pixel_original[1])
            # Convert to camera coordinates
            x = (max_pixel_original[1] - cx)/(fx) * point_depth
            y = (max_pixel_original[0] - cy)/(fy) * point_depth
            z = point_depth
            
            # # DEBUG
            # print("Point depth:", point_depth)
            # temp_x = 1/(fy) * point_depth
            # temp_y = 1/(fx) * point_depth
            # print("temp camera coordinates (x,y):", temp_x, temp_y)
            
            if np.isnan(z):
                return

            # Add visualization of the calculated grasp point on the original image
            if visualize_crop:
                # Make a copy of the original visualization
                grasp_point_viz = np.copy(depth_viz_color)
                
                # Draw grasp point on original image
                cv2.circle(grasp_point_viz, (max_pixel_original[1], max_pixel_original[0]), 5, (0, 255, 0), -1)
                
                # Draw grasp orientation line
                length = 25
                end_x = int(max_pixel_original[1] + length * np.cos(ang))
                end_y = int(max_pixel_original[0] + length * np.sin(ang))
                cv2.line(grasp_point_viz, (max_pixel_original[1], max_pixel_original[0]), 
                        (end_x, end_y), (0, 255, 0), 2)
                
                # Add info text
                cv2.putText(grasp_point_viz, f"Grasp point", 
                            (max_pixel_original[1]+10, max_pixel_original[0]-10), 
                            font, 0.5, (0, 255, 0), 1)
                
                # Display visualization with grasp point
                cv2.imshow("Original With Grasp", grasp_point_viz)
                cv2.waitKey(1)

        with TimeIt('Draw'):
            # Draw grasp markers on the points_out and publish it
            grasp_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            
            # Convert to tensor for better error handling
            points_out_tensor = torch.from_numpy(points_out)
            grasp_img[:,:,2] = (points_out_tensor * 255.0).cpu().numpy().astype(np.uint8)

            # Draw circle at maximum point
            rr, cc = disk(self.prev_mp, 5, shape=(target_size, target_size, 3))
            grasp_img[rr, cc, 0] = 0
            grasp_img[rr, cc, 1] = 255
            grasp_img[rr, cc, 2] = 0

            # Draw rectangle showing grasp orientation and width
            # print("Drawing rectangle at:", max_pix_target[1], max_pix_target[0], "with angle:", ang, "and width:", width)
            # grasp_img = self.draw_rectangle(max_pix_target[1], max_pix_target[0], ang, width, grasp_img)

        return [x, y, z, ang, width, depth_center], grasp_img