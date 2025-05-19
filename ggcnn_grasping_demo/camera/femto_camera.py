import math
import numpy as np
import cv2
import time
from pyorbbecsdk import *  # Using the Orbbec SDK for Femtbo Bolt camera

class CameraIntrinsics: # class to format the cam instrinsics
    def __init__(self, fx, fy, cx, cy, width, height):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
    
    def __str__(self):
        return f"CameraIntrinsics(fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, width={self.width}, height={self.height})"


class FemtboBoltCamera(object):
    def __init__(self, color_wh = [1280,960], depth_wh = [1024,1024]):
        self.pipeline = Pipeline()
        self.config = Config()
        
        try:
            # Set up color stream with default profile
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            self.color_profile = profile_list.get_video_stream_profile(color_wh[0], color_wh[1], OBFormat.RGB, 15)
            self.config.enable_stream(self.color_profile)
            
            # Set up depth stream with default profile
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            self.depth_profile = profile_list.get_video_stream_profile(depth_wh[0], depth_wh[1], OBFormat.Y16, 15)
            self.config.enable_stream(self.depth_profile)
            
            # Get original intrinsics and extrinsics directly from the profiles
            self.color_intrinsics = self.color_profile.get_intrinsic()
            self.depth_intrinsics = self.depth_profile.get_intrinsic()
            self.color_distortion = self.color_profile.get_distortion()
            self.depth_distortion = self.depth_profile.get_distortion()
            
            # Get extrinsic parameters (depth to color)
            self.extrinsic = self.depth_profile.get_extrinsic_to(self.color_profile)
            
            # Enable frame synchronization
            self.pipeline.enable_frame_sync()
            
            # Start the pipeline
            self.pipeline.start(self.config)
            
            # Create align filter to align depth to color
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            
            # Initialize updated intrinsics
            self.aligned_color_intrinsics = None
            self.aligned_depth_intrinsics = None
            
            print("\nOriginal Color Intrinsics:", self.color_intrinsics)
            print("Original Depth Intrinsics:", self.depth_intrinsics)
            # print("Extrinsic Parameters:", self.extrinsic)
            
        except Exception as e:
            print(f"Initialization error: {e}")
            raise
            
        # Set depth range values
        self.MIN_DEPTH = 20  # 20mm
        self.MAX_DEPTH = 10000  # 10000mm
        # calculate aligned instrincs by default and account for the slower frame transfer
        while self.aligned_depth_intrinsics is None:
            self.get_frames()
        
    def stop(self):
        """Stop the camera pipeline"""
        self.pipeline.stop()
    
    def get_frames(self):
        """Get synchronized and aligned frames"""
        try:
            frames = self.pipeline.wait_for_frames(1000 // 15 * 3)  # 3x the expected frame period - 15fps to allow for timesync
            if not frames:
                print("franes not received before timeout")
                return None
                
            # Get original color and depth frames
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                print("num of color frames didn't match the depth frames")
                return None
                
            # Align frames - this aligns depth to color space
            aligned_frames = self.align_filter.process(frames)
            if not aligned_frames:
                print("frame alignment failed")
                return None
                
            # Convert to frame set
            aligned_frames = aligned_frames.as_frame_set()
            
            # Get aligned color and depth frames
            aligned_color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            
            if not aligned_color_frame or not aligned_depth_frame:
                print("num of colored and depth frames after alignment didn;t match")
                return None
                
            # Update intrinsics for aligned frames if not done yet
            if self.aligned_color_intrinsics is None or self.aligned_depth_intrinsics is None:
                # print("Calling alingned instrinsic calculation")
                self._update_aligned_intrinsics(aligned_color_frame, aligned_depth_frame)
            
            return aligned_frames
            
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None
    
    def get_intrinsics(self):
        return self.color_intrinsics, self.aligned_depth_intrinsics

    def _update_aligned_intrinsics(self, aligned_color_frame, aligned_depth_frame):
        """Calculate updated intrinsics after alignment"""
        try:
            # print("Calculating aligned intrinsics")
            # After alignment, the depth frame is aligned to the color frame's space
            # For the color frame, intrinsics remain the same
            self.aligned_color_intrinsics = self.color_intrinsics
            
            # For depth frame, we need to consider the transformation
            # In theory, after alignment, the depth frame uses the color intrinsics
            # but with possibly different dimensions
            depth_width = aligned_depth_frame.get_width()
            depth_height = aligned_depth_frame.get_height()
            
            # Calculate scale factors if dimensions changed
            width_scale = depth_width / self.color_intrinsics.width
            height_scale = depth_height / self.color_intrinsics.height
            
            # After alignment, depth frame intrinsics should be similar to color
            # but possibly with scaled parameters if the resolution is different
            self.aligned_depth_intrinsics = CameraIntrinsics(
                fx=self.color_intrinsics.fx * width_scale,
                fy=self.color_intrinsics.fy * height_scale,
                cx=self.color_intrinsics.cx * width_scale,
                cy=self.color_intrinsics.cy * height_scale,
                width=depth_width,
                height=depth_height
            )
            
            print("\nUpdated Intrinsics after Alignment:")
            print("Aligned Color Intrinsics: Same as original")
            print(f"Aligned Depth Intrinsics: {self.aligned_depth_intrinsics}")
                
        except Exception as e:
            print(f"Error updating aligned intrinsics: {e}")
    
    def get_camera_parameters(self):
        """Get all camera parameters including aligned intrinsics"""
        return {
            'color_intrinsics': self.color_intrinsics,
            'depth_intrinsics': self.depth_intrinsics,
            'aligned_color_intrinsics': self.aligned_color_intrinsics,
            'aligned_depth_intrinsics': self.aligned_depth_intrinsics,
            'color_distortion': self.color_distortion,
            'depth_distortion': self.depth_distortion,
            'extrinsic': self.extrinsic
        }
    
    def get_images(self):
        """Get aligned color and depth images"""
        frames = self.get_frames()
        if not frames:
            return None, None
            
        aligned_color_frame = frames.get_color_frame()
        aligned_depth_frame = frames.get_depth_frame()
        
        if not aligned_color_frame or not aligned_depth_frame:
            return None, None
            
        # Convert color frame to image
        color_image = frame_to_bgr_image(aligned_color_frame)
        if color_image is None:
            print("Failed to convert color frame to image")
            return None, None
            
        # Convert depth frame to image
        try:
            # Get depth data as numpy array
            depth_data = np.frombuffer(aligned_depth_frame.get_data(), dtype=np.uint16).reshape(
                (aligned_depth_frame.get_height(), aligned_depth_frame.get_width()))
                
            # Apply depth scale
            depth_scale = aligned_depth_frame.get_depth_scale()
            depth_image = depth_data.astype(np.float32) * depth_scale
            
        except Exception as e:
            print(f"Error processing depth image: {e}")
            return color_image, None
            
        return color_image, depth_image


# Helper function to convert frame to image
def frame_to_bgr_image(frame):
    """Convert a VideoFrame to a BGR image"""
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        # This would require specific I420 to BGR conversion
        print("I420 format not fully supported")
        return None
    elif color_format == OBFormat.NV12:
        # This would require specific NV12 to BGR conversion
        print("NV12 format not fully supported")
        return None
    elif color_format == OBFormat.NV21:
        # This would require specific NV21 to BGR conversion
        print("NV21 format not fully supported")
        return None
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format:", color_format)
        return None
        
    return image


if __name__ == '__main__':
    try:
        # Create camera object
        cam = FemtboBoltCamera()
        
        # Wait a bit for camera to initialize
        import time
        time.sleep(1)
        
        # Get camera parameters
        params = cam.get_camera_parameters()
        
        # Main display loop
        while True:
            # Get aligned images
            color_image, depth_image = cam.get_images()
            
            if color_image is not None and depth_image is not None:
                # Display color image
                cv2.imshow('COLOR', color_image)
                
                # Normalize and colorize depth for display
                depth_display = np.nan_to_num(depth_image).astype(np.uint16)
                depth_display = cv2.normalize(depth_display, None, 0, 255, cv2.NORM_MINMAX)
                depth_colormap = cv2.applyColorMap(depth_display.astype(np.uint8), cv2.COLORMAP_JET)
                
                # Display depth image
                cv2.imshow('DEPTH', depth_colormap)
                
                # Optional: Display overlay
                overlay = cv2.addWeighted(color_image, 0.7, depth_colormap, 0.3, 0)
                cv2.imshow('OVERLAY', overlay)
            
            # Check for key press
            key = cv2.waitKey(1)
            # Press 'q' or ESC to close the image window
            if key & 0xFF == ord('q') or key == 27:
                break
                
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # Ensure camera is properly stopped
        if 'cam' in locals():
            cam.stop()
        cv2.destroyAllWindows()