import sys
import numpy as np
import depthai as dai
import cv2
import math


class DepthAiCamera(object):
    def __init__(self, width=640, height=400, fps=30, disable_rgb=False):
        np.set_printoptions(threshold=sys.maxsize) # print all data in an image
        # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
        extended_disparity = True
        # Better accuracy for longer distance, fractional disparity 32-levels:
        subpixel = False
        # Better handling for occlusions:
        lr_check = True

        # Create pipeline
        pipeline = dai.Pipeline()
        self.disable_rgb = disable_rgb

        # Define sources and outputs
        if not self.disable_rgb:
            camRgb = pipeline.create(dai.node.ColorCamera)
            xoutRgb = pipeline.create(dai.node.XLinkOut)
            xoutRgb.setStreamName("rgb")
            # Properties
            camRgb.setPreviewSize(width, height)
            camRgb.setInterleaved(False)
            camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
            camRgb.setFps(fps)
            # Linking
            camRgb.preview.link(xoutRgb.input)

        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)
        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        # Properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P) # 640*400
        monoLeft.setFps(fps)
        monoLeft.setCamera("left")
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P) # 640*400
        monoRight.setFps(fps)
        monoRight.setCamera("right")

        # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.initialConfig.setDisparityShift(30)
        depth.setLeftRightCheck(lr_check)
        depth.initialConfig.setLeftRightCheckThreshold(15)
        depth.setExtendedDisparity(True) 
        depth.setSubpixel(subpixel)
        # depth.initialConfig.setDisparityShift(30) #optional: set disparity shift to enhance close observation
        depth.initialConfig.setConfidenceThreshold(200)
        # Linking
        monoLeft.out.link(depth.left)
        monoRight.out.link(depth.right)
        depth.depth.link(xoutDepth.input) # depth unit: mm, max: 65535
        # print(depth.initialConfig.getMaxDisparity()) # 190.0 if setExtendedDisparity(true)

        self.width = width
        self.height = height
        self.pipeline = pipeline
        self.depth = depth
        self.device = dai.Device(self.pipeline)
        # self.device.setIrLaserDotProjectorBrightness(100)
        # self.device.setIrFloodLightBrightness(100) 
        # Output queue will be used to get the disparity frames from the outputs defined above
        if not self.disable_rgb:
            self.queRgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        else:
            self.queRgb = None
        self.queDepth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    
    def __exit__(self):
        self.device.close()

    def get_intrinsics(self):
        calibData = self.device.readCalibration()
        # M_rgb, width, height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_A)
        # M_left, width, height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_B)
        # M_Right, width, height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_C)

        M_rgb = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, self.width, self.height)
        M_left = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, self.width, self.height)
        M_Right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, self.width, self.height)
        # M_Right = calibData.getCameraIntrinsics(calibData.getStereoRightCameraId(), self.width, self.height)

        # M_depth = (np.array(M_left) + np.array(M_Right)) / 2
        M_depth = np.array(M_left)
        return np.array(M_rgb), M_depth

        # R1 = np.array(calibData.getStereoLeftRectificationRotation())
        # R2 = np.array(calibData.getStereoRightRectificationRotation())

        # H_left = np.matmul(np.matmul(M_Right, R1), np.linalg.inv(M_left))
        # print("LEFT Camera stereo rectification matrix...")
        # print(H_left)

        # H_right = np.matmul(np.matmul(M_Right, R1), np.linalg.inv(M_Right))
        # print("RIGHT Camera stereo rectification matrix...")
        # print(H_right)

    def get_frames(self):
        if self.queRgb is not None:
            inRgb = self.queRgb.get()  # blocking call, will wait until a new data has arrived
            colorFrame = inRgb.getCvFrame()
        else:
            colorFrame = None
        inDepth = self.queDepth.get()  # blocking call, will wait until a new data has arrived
        depthFrame = inDepth.getFrame()
        # Normalization for better visualization
        # depthFrame = (depthFrame * (255 / self.depth.initialConfig.getMaxDisparity())).astype(np.uint8) # for better visualization
        # print(depthFrame) # range 0-65535
        # cv2.imshow("depth", depthFrame.astype(np.uint8))
        # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        # depthFrame = cv2.applyColorMap(depthFrame, cv2.COLORMAP_JET) # input to colorMap must be 0-255
        # cv2.imshow("disparity_color", depthFrame)
        return colorFrame, depthFrame
    
    def get_images(self):
        frames = self.get_frames()
        depth_image = np.asanyarray(frames[1]) * 0.001
        depth_image[depth_image == 0] = math.nan
        return frames[0], depth_image
        # return frames[0], np.asanyarray(frames[1]) * 0.001 # frames[1].astype(np.uint8)


if __name__ == '__main__':
    cam = DepthAiCamera()
    color_intrin, depth_intrin = cam.get_intrinsics()
    print(color_intrin)
    print(depth_intrin)
    fx = depth_intrin[0][0]
    fy = depth_intrin[0][2]
    cx = depth_intrin[1][1]
    cy = depth_intrin[1][2]
    print(fx, fy, cx, cy)

    while True:
        color_image, depth_image = cam.get_images()
        if color_image is not None:
            cv2.imshow('COLOR', color_image) # 显示彩色图像
        cv2.imshow('DEPTH', depth_image) # 显示深度图像

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cam.stop()
            break