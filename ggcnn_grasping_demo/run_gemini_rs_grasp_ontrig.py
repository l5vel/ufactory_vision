import sys
import time
import signal
from threading import Thread, Event

# Import the core grasping functionality
from run_rs_grasp import perform_grasp
from gemini_bbox import GeminiBbox
from run_rs_grasp_ontrig import NetworkTablesGraspTrigger
# Setup logging for NetworkTables

ARM_IP = '172.16.0.13'
RASPI_IP = '172.16.0.12'

class GeminiGGCNNGrasp:
    def __init__(self, arm_ip=ARM_IP, raspi_ip=RASPI_IP):
        self.gemini_bbox_obj = GeminiBbox()
        self.nt_obj = None
        self.grasp_trigger = None
        self.arm_ip = arm_ip
        self.raspi_ip = raspi_ip

    def initialize(self):
        # Initialize the NetworkTables grasp trigger
        self.nt_obj = self.grasp_nt_init()

        # Initialize the GeminiBbox object
        self.gemini_bbox_obj = self.gemini_init()

    def run_gemini_detect(self, img, query):
        return self.gemini_bbox_obj.run_gemini_detect(img, query)
    
    def gemini_init(self):
        # Initialize the GeminiBbox object
        gemini_bbox_obj = GeminiBbox()
        return gemini_bbox_obj

    def grasp_nt_init(self):
        # init nt with raspi and grasp with arm and realsense
        nt_obj = NetworkTablesGraspTrigger(self.raspi_ip,self.arm_ip)
        nt_obj.initialize_networktables()
        nt_obj.start_monitoring()
        return nt_obj

    def run(self):
        # Main loop to run the grasping process
        while True:
            # Wait for the trigger signal from NetworkTables
            if self.nt_obj.is_triggered():
                # Get the image from the camera
                img = self.nt_obj.get_image()

                # Perform object detection using Gemini
                detection = self.run_gemini_detect(img,)

                # If detections are found, perform grasping
                if detection:
                    label = detection['label']
                    bbox = detection['bounding_box']
                    print(f"Detected {label} with bounding box: {bbox}")
                    # Perform grasping using the detected bounding box
                    perform_grasp(img, bbox)

                # Reset the trigger signal
                self.nt_obj.reset_trigger()

def main():
        gemini_bbox_obj = gemini_init()
        nt_obj = grasp_nt_init()


        object_to_find = "banana"

        detection_prompt = f"""
        Analyze the image and identify all instances of '{object_to_find}'.
        For each '{object_to_find}' found, output its bounding box coordinates in the format:
        {object_to_find} : (x_min, y_min, x_max, y_max)
        Here, (x_min, y_min) represents the top-left corner of the bounding box, and (x_max, y_max) represents the bottom-right corner.
        If multiple '{object_to_find}' are detected, list each one on a separate line in this exact format, without any additional text or bullet points.
        """
        highc_detection_prompt = f"""
        Analyze the image and identify any instances of '{object_to_find}'.
        From these, determine which '{object_to_find}' instance you are most confident about (the clearest, most certain detection).
        Output **only** the bounding box coordinates of that single most confident '{object_to_find}' in the format:
        {object_to_find} : (x_min, y_min, x_max, y_max)
        (Note: (x_min, y_min) is the top-left corner and (x_max, y_max) is the bottom-right corner of the bounding box.)
        If no '{object_to_find}' is detected in the image, clearly output "No {object_to_find} found."
        """
        
        gemini_bbox_obj.run_gemini_detect(img, highc_detection_prompt)

if __name__ == '__main__':
    sys.exit(main())