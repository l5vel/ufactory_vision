# ufactory\_vision

## Project Overview

`ufactory_vision` is a vision-based grasping demo based on UFACTORY 850 and xArm Serials with Intel Realsense D435 camera or Luxonis OAK-D-Pro-PoE depth camera. Users can quickly implement vision-based object detection and grasping with this project.

[![Watch the video](https://img.youtube.com/vi/ijnuqsNcfUY/0.jpg)](https://www.youtube.com/watch?v=ijnuqsNcfUY)





## Hardware Requirements


**UFACTORY Robot with Intel Realsense D435 Camera** 

We used UFACTORY xArm 6 robot in the video, UFACTORY 850 and xArm 5, xArm 7 also works. Lite 6 does not support.

- **Robot Arm**: [UFACTORY xArm 6](https://www.ufactory.cc/products/xarm)
- **Gripper**: [UFACTORY xArm Gripper](https://www.ufactory.cc/product-page/ufactory-xarm-gripper/)
- **Camera**: [Intel RealSense D435](https://www.intelrealsense.com/depth-camera-d435/)
- **Camera Mount**: Provided by UFACTORY (available for purchase or 3D printing)
  - Purchase Link: [UFACTORY Camera Stand](https://www.ufactory.cc/product-page/ufactory-xarm-camera-stand/)
  - 3D File Download: [Camera\_Stand\_1300.STEP](https://www.ufactory.cc/wp-content/uploads/2024/05/CameraStand_1300.zip) 



**UFACTORY Robot with Luxonis OAK-D-Pro-PoE Camera**

We used a customized 850 with built-in gigabit net cable to shoot the video, [contact UFACTORY sales](https://www.ufactory.cc/contact-us/) if you need the customized gigabit version UFACTORY 850 robot. UFACTORY xArm 5, xArm 6 and xArm 7 also work, Lite 6 does not support.

- **Robot Arm**: [UFACTORY 850](https://www.ufactory.cc/ufactory-850/)[Network Cable outside the Robot]
- **Gripper**: [UFACTORY xArm Gripper](https://www.ufactory.cc/product-page/ufactory-xarm-gripper/)
- **Camera**: [OAK-D Pro PoE](https://shop.luxonis.com/products/oak-d-pro-poe?variant=42469208883423)

- **PoE Injector**: Tenda POE30G-AT, 1000M,30W(Other brand will also work)
- **Camera Mount**: Provided by UFACTORY (3D printing or CNC with 3D file)
  - 3D File Download: [Camera\_Stand\_850.STEP](Wait for upload) 


## Sofeware 

### Supported Python Versions

Supported Python versions: 3.8-3.11 (Recommended: 3.11).

## Installation

### Clone the repository

```bash
git clone https://github.com/xArm-Developer/ufactory_vision.git
```

### Create a Python Virtual Environment

It is recommended to use a virtual environment for running this project.

#### **Windows (Using Anaconda)**

```bash
conda create --name ufactory_vision python=3.11
conda activate ufactory_vision
```

#### **Linux (Using venv)**

```bash
python3.11 -m venv ufactory_vision
source ufactory_vision/bin/activate
```

### Install dependencies

**Intel Realsense D435 Camera**
```bash
cd ggcnn_grasping_demo
pip install -r requirements_rs.txt
```
**Luxonis OAK-D-Pro-PoE Camera**
```bash
cd ggcnn_grasping_demo
pip install -r requirements_depthai.txt
```

## Run Example 

**Intel Realsense D435 Camera**

```bash
python run_rs_grasp.py 192.168.1.xxx
```

**Luxonis OAK-D-Pro-PoE Camera**

```bash
python run_depthai_grasp.py 192.168.1.xxx
```


Replace `192.168.1.xxx` with the IP address of the controller.

## Important Notes

Before running the example, ensure that collision detection is enabled. It is recommended to set the collision sensitivity to 3 or higher.

## License

This project is licensed under the **BSD 3-Clause License**. For details, please check the [LICENSE](LICENSE) file.

## Reference

We built the demo base on below opensource projects.

[GGCNN](https://github.com/dougsm/ggcnn) 

[ggcnn_kinova_grasping](https://github.com/dougsm/ggcnn_kinova_grasping)
