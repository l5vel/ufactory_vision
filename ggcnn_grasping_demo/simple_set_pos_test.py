from xarm.wrapper import XArmAPI
import time

TARGET_POS = [693.7614905065897, -62.06180540611822, 222.74105378615516, 47.22492950695822, -84.07915, 5.0579]
ARM_IP = '172.16.0.13'
def main():
    # Replace with your robot's IP address
    robot_ip = ARM_IP  
    
    # Initialize the robot
    arm = XArmAPI(robot_ip)
    
    # Clean any errors and set up the robot
    arm.clean_error()
    arm.clean_warn()
    time.sleep(1)
    arm.motion_enable(True)
    arm.set_mode(0)  # Position control mode
    arm.set_state(0)  # Ready state
    
    # Define the target position [x, y, z, roll, pitch, yaw]
    # Units: mm for position, degrees for orientation
    target_position = TARGET_POS
    
    # Move to the target position
    print(f"Moving to position: {target_position}")
    arm.set_position(
        x=target_position[0], 
        y=target_position[1], 
        z=target_position[2],
        roll=target_position[3], 
        pitch=target_position[4], 
        yaw=target_position[5],
        speed=100,  # Speed in mm/s
        wait=True   # Wait for the movement to complete
    )
    
    print("Movement completed")
    
    # Disconnect from the robot
    arm.disconnect()

if __name__ == "__main__":
    main()