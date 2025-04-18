import sys
import time
import cv2
import torch
import numpy as np
import scipy.ndimage as ndimage
from skimage.draw import disk
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
        self.model = torch.load('models/ggcnn_epoch_23_cornell', map_location=self.device, weights_only=False)
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

    def get_grasp_img_precropped(self, depth_crop, cx, cy, fx, fy):
        
        with TimeIt('Inference'):
            # Run it through the network. Normalize:
            depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
            # target: REMOVE! #############
            input_numpy_array = depth_crop.reshape((1, 300, 300, 1)) # numpy ndaray.reshape(new_shape), 1*300*300*1 must be equal to total element number!
            # input_tensor = torch.Tensor(input_numpy_array) # Deep copy
            input_tensor = torch.as_tensor(input_numpy_array).to(self.device) # shallow copy
            # with graph.as_default():
            #     pred_out = model.predict(depth_crop.reshape((1, 300, 300, 1)))
            input_tensor = input_tensor.permute(0, 3, 1, 2)
            with torch.no_grad():
                pred_out = self.model(input_tensor)

            # output as 4 300*300 images?
            # output_numpy = output_torch.numpy()
            # yy[0].cpu().numpy().squeeze().shape
            points_out = pred_out[0].cpu().numpy().squeeze() # shape is now (300,300)
        
        with TimeIt('Trig'):
            # Calculate the angle map.
            cos_out = pred_out[1].cpu().numpy().squeeze()
            sin_out = pred_out[2].cpu().numpy().squeeze()
            ang_out = np.arctan2(sin_out, cos_out)/2.0

            width_out = pred_out[3].cpu().numpy().squeeze() * 150.0  # Scaled 0-150:0-1

        with TimeIt('Filter'):
            # Filter the outputs.
            # points_out = ndimage.filters.gaussian_filter(points_out, 5.0)  # 3.0
            # ang_out = ndimage.filters.gaussian_filter(ang_out, 2.0)
            # use ndimage.gaussian_filter replace ndimage.filters.gaussian_filter
            points_out = ndimage.gaussian_filter(points_out, 5.0)  # 3.0
            ang_out = ndimage.gaussian_filter(ang_out, 2.0)
        
        with TimeIt('Control'):
            # Calculate the best pose from the camera intrinsics.
            maxes = None

            ALWAYS_MAX = False  # Use ALWAYS_MAX = True for the open-loop solution.

            if self.robot_z > 0.34 or ALWAYS_MAX:  # > 0.34 initialises the max tracking when the robot is reset.
                # Track the global max.
                max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
                # self.prev_mp = max_pixel.astype(np.int)
                # use np.int64 replace np.int
                self.prev_mp = max_pixel.astype(np.int64)
                # print(points_out[max_pixel[0],max_pixel[1]])
            else:
                # Calculate a set of local maxes.  Choose the one that is closes to the previous one.
                maxes = peak_local_max(points_out, min_distance=10, threshold_abs=0.1, num_peaks=3)
                if maxes.shape[0] == 0:
                    return
                max_pixel = maxes[np.argmin(np.linalg.norm(maxes - self.prev_mp, axis=1))]

                # Keep a global copy for next iteration.
                # self.prev_mp = (max_pixel * 0.25 + self.prev_mp * 0.75).astype(np.int)
                # use np.int64 replace np.int
                self.prev_mp = (max_pixel * 0.25 + self.prev_mp * 0.75).astype(np.int64)

            # print("max_pixel: 300*300 [%d, %d]"%(max_pixel[0], max_pixel[1]))
            ang = ang_out[max_pixel[0], max_pixel[1]]
            width = width_out[max_pixel[0], max_pixel[1]]
            # Convert max_pixel back to uncropped/resized image coordinates in order to do the camera transform.
            max_pixel = ((np.array(max_pixel) / 300.0 * crop_size) + np.array([height_crop // 2, width_crop // 2]))
            # max_pixel = np.round(max_pixel).astype(np.int)
            # use np.int64 replace np.int
            max_pixel = np.round(max_pixel).astype(np.int64)
            # print("max_pixel: original [%d, %d]"%(max_pixel[0], max_pixel[1]))
            point_depth = depth_image[max_pixel[0], max_pixel[1]]
            # These magic numbers are my camera intrinsic parameters. # in camera coordinate system
            x = (max_pixel[1] - cx)/(fx) * point_depth
            y = (max_pixel[0] - cy)/(fy) * point_depth
            z = point_depth
            # print("converted xyz in cam coordinate: [%lf, %lf, %lf]"%(x,y,z))
            if np.isnan(z):
                return

        with TimeIt('Draw'):
            # Draw grasp markers on the points_out and publish it. (for visualisation)
            grasp_img = np.zeros((300, 300, 3), dtype=np.uint8)
            # direct assignment is giving casting errors
            # Convert the NumPy array to a PyTorch Tensor
            points_out_tensor = torch.from_numpy(points_out)
            # print("Are there NaNs in points_out?", torch.isnan(points_out_tensor).any())
            # print("Are there Infs in points_out?", torch.isinf(points_out_tensor).any())
            # print("Min value of points_out (Tensor):", torch.min(points_out_tensor))
            # print("Max value of points_out (Tensor):", torch.max(points_out_tensor))
            grasp_img[:,:,2] = (points_out_tensor * 255.0).cpu().numpy().astype(np.uint8)

            # grasp_img_plain = grasp_img.copy()

            # use disk replace circle
            # rr, cc = circle(self.prev_mp[0], self.prev_mp[1], 5)
            rr, cc = disk(self.prev_mp, 5, shape=(300, 300, 3))
            grasp_img[rr, cc, 0] = 0
            grasp_img[rr, cc, 1] = 255
            grasp_img[rr, cc, 2] = 0
        
        depth_center = 0 # will be assigned in the run method
        return [x, y, z, ang, width, depth_center], grasp_img

    def get_grasp_img(self, depth_image, cx, cy, fx, fy, robot_z=None):
        if robot_z is not None:
            self.robot_z = robot_z
        with TimeIt('Crop'):
            # Crop a square (400*400) out of the middle of the depth and resize it to 300*300
            crop_size = 400
            height_crop = max(0, self.height - crop_size)
            width_crop = max(0, self.width - crop_size)
            # depth_crop is from a copy of original depth cv image from ROS msg
            depth_crop = cv2.resize(depth_image[height_crop//2:height_crop//2+crop_size, width_crop//2:width_crop//2+crop_size], (300, 300))
            # Replace nan with 0 for inpainting.
            depth_crop = depth_crop.copy() # depth_crop: array([ 1.,  2.,  3., nan])
            depth_nan = np.isnan(depth_crop).copy() # depth_nan: array([False, False, False,  True])
            depth_crop[depth_nan] = 0 # ??? depth_crop: array([ 1.,  2.,  3., 0.0])

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
        
        with TimeIt('Calculate Depth'): # ???
            # Figure out roughly the depth in mm of the part between the grippers for collision avoidance.
            depth_center = depth_crop[100:141, 130:171].flatten()
            depth_center.sort()
            depth_center = depth_center[:10].mean() * 1000.0
        
        with TimeIt('Inference'):
            # Run it through the network. Normalize:
            depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
            # target: REMOVE! #############
            input_numpy_array = depth_crop.reshape((1, 300, 300, 1)) # numpy ndaray.reshape(new_shape), 1*300*300*1 must be equal to total element number!
            # input_tensor = torch.Tensor(input_numpy_array) # Deep copy
            input_tensor = torch.as_tensor(input_numpy_array).to(self.device) # shallow copy
            # with graph.as_default():
            #     pred_out = model.predict(depth_crop.reshape((1, 300, 300, 1)))
            input_tensor = input_tensor.permute(0, 3, 1, 2)
            with torch.no_grad():
                pred_out = self.model(input_tensor)

            # output as 4 300*300 images?
            # output_numpy = output_torch.numpy()
            # yy[0].cpu().numpy().squeeze().shape
            points_out = pred_out[0].cpu().numpy().squeeze() # shape is now (300,300)
            points_out[depth_nan] = 0
        
        with TimeIt('Trig'):
            # Calculate the angle map.
            cos_out = pred_out[1].cpu().numpy().squeeze()
            sin_out = pred_out[2].cpu().numpy().squeeze()
            ang_out = np.arctan2(sin_out, cos_out)/2.0

            width_out = pred_out[3].cpu().numpy().squeeze() * 150.0  # Scaled 0-150:0-1

        with TimeIt('Filter'):
            # Filter the outputs.
            # points_out = ndimage.filters.gaussian_filter(points_out, 5.0)  # 3.0
            # ang_out = ndimage.filters.gaussian_filter(ang_out, 2.0)
            # use ndimage.gaussian_filter replace ndimage.filters.gaussian_filter
            points_out = ndimage.gaussian_filter(points_out, 5.0)  # 3.0
            ang_out = ndimage.gaussian_filter(ang_out, 2.0)
        
        with TimeIt('Control'):
            # Calculate the best pose from the camera intrinsics.
            maxes = None

            ALWAYS_MAX = False  # Use ALWAYS_MAX = True for the open-loop solution.

            if self.robot_z > 0.34 or ALWAYS_MAX:  # > 0.34 initialises the max tracking when the robot is reset.
                # Track the global max.
                max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
                # self.prev_mp = max_pixel.astype(np.int)
                # use np.int64 replace np.int
                self.prev_mp = max_pixel.astype(np.int64)
                # print(points_out[max_pixel[0],max_pixel[1]])
            else:
                # Calculate a set of local maxes.  Choose the one that is closes to the previous one.
                maxes = peak_local_max(points_out, min_distance=10, threshold_abs=0.1, num_peaks=3)
                if maxes.shape[0] == 0:
                    return
                max_pixel = maxes[np.argmin(np.linalg.norm(maxes - self.prev_mp, axis=1))]

                # Keep a global copy for next iteration.
                # self.prev_mp = (max_pixel * 0.25 + self.prev_mp * 0.75).astype(np.int)
                # use np.int64 replace np.int
                self.prev_mp = (max_pixel * 0.25 + self.prev_mp * 0.75).astype(np.int64)

            # print("max_pixel: 300*300 [%d, %d]"%(max_pixel[0], max_pixel[1]))
            ang = ang_out[max_pixel[0], max_pixel[1]]
            width = width_out[max_pixel[0], max_pixel[1]]
            # Convert max_pixel back to uncropped/resized image coordinates in order to do the camera transform.
            max_pixel = ((np.array(max_pixel) / 300.0 * crop_size) + np.array([height_crop // 2, width_crop // 2]))
            # max_pixel = np.round(max_pixel).astype(np.int)
            # use np.int64 replace np.int
            max_pixel = np.round(max_pixel).astype(np.int64)
            # print("max_pixel: original [%d, %d]"%(max_pixel[0], max_pixel[1]))
            point_depth = depth_image[max_pixel[0], max_pixel[1]]
            # These magic numbers are my camera intrinsic parameters. # in camera coordinate system
            x = (max_pixel[1] - cx)/(fx) * point_depth
            y = (max_pixel[0] - cy)/(fy) * point_depth
            z = point_depth
            # print("converted xyz in cam coordinate: [%lf, %lf, %lf]"%(x,y,z))
            if np.isnan(z):
                return

        with TimeIt('Draw'):
            # Draw grasp markers on the points_out and publish it. (for visualisation)
            grasp_img = np.zeros((300, 300, 3), dtype=np.uint8)
            # direct assignment is giving casting errors
            # Convert the NumPy array to a PyTorch Tensor
            points_out_tensor = torch.from_numpy(points_out)
            # print("Are there NaNs in points_out?", torch.isnan(points_out_tensor).any())
            # print("Are there Infs in points_out?", torch.isinf(points_out_tensor).any())
            # print("Min value of points_out (Tensor):", torch.min(points_out_tensor))
            # print("Max value of points_out (Tensor):", torch.max(points_out_tensor))
            grasp_img[:,:,2] = (points_out_tensor * 255.0).cpu().numpy().astype(np.uint8)

            # grasp_img_plain = grasp_img.copy()

            # use disk replace circle
            # rr, cc = circle(self.prev_mp[0], self.prev_mp[1], 5)
            rr, cc = disk(self.prev_mp, 5, shape=(300, 300, 3))
            grasp_img[rr, cc, 0] = 0
            grasp_img[rr, cc, 1] = 255
            grasp_img[rr, cc, 2] = 0
        
        return [x, y, z, ang, width, depth_center], grasp_img