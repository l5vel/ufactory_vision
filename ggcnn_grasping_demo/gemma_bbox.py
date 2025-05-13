import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import time
import requests
import base64
from io import BytesIO

# Import camera module
from camera.rs_camera import RealSenseCamera

class Gemma3Bbox:
    def __init__(self, ollama_url="http://10.94.105.101:11434/api/generate"):
        """
        Initializes the Gemma3Bbox class using Ollama's Gemma3 27B multimodal model.

        Args:
            ollama_url (str): The URL of the Ollama API endpoint.
        """
        self.ollama_url = ollama_url
        # self.model_name = "gemma3:27b"
        # self.model_name = "llama3.2-vision"
        self.model_name = "llava:34b"
        
        # Regular expression to find labels before ':' and coordinates inside '()'
        self.bbox_pattern = r"(.+?):\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)"

        # Turn on interactive mode for visualization
        plt.ion()
        # Create a persistent figure and axes
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        
        # Test connection to Ollama
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server"""
        try:
            # Simple ping to check if the server is reachable
            response = requests.get(self.ollama_url.replace("/api/generate", "/api/tags"))
            if response.status_code == 200:
                print(f"Successfully connected to Ollama server")
            else:
                print(f"Warning: Ollama server returned status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to Ollama server: {e}")
    
    def encode_image_base64(self, image_pil):
        """
        Encode a PIL image to base64 string for Ollama API input.
        """
        buffer = BytesIO()
        image_pil.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str
    
    def parse_response(self, text_response):
        """
        Parse the text response from Gemma3 to extract object labels and bounding boxes.
        """
        # Find all matches in the extracted text
        matches = re.findall(self.bbox_pattern, text_response)
        
        # Process the matches
        detections = []
        for label, x_min, y_min, x_max, y_max in matches:
            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
            detections.append({"label": label.strip(), "bounding_box": bbox})
        
        return detections
    
    def detect_objects(self, img, query):
        """
        Detect objects in the image using Gemma3 multimodal model via Ollama.
        """
        try:
            # Encode the image to base64
            img_base64 = self.encode_image_base64(img)
            
            # Prepare the request payload for Ollama
            payload = {
                "model": self.model_name,
                "prompt": query,
                "images": [img_base64],
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Low temperature for more precise outputs
                    "top_p": 0.9
                }
            }
            
            # Make the API request to Ollama
            response = requests.post(self.ollama_url, json=payload)
            
            if response.status_code != 200:
                print(f"Error: Ollama API returned status code {response.status_code}")
                print(f"Response content: {response.text}")
                return None
            
            # Extract the response text
            response_json = response.json()
            response_text = response_json.get("response", "")
            
            # Parse the response to extract bounding boxes
            temp_detect = self.parse_response(response_text)
            
            # Process the detections to normalize coordinates
            detections = []
            for detection in temp_detect:
                bbox = detection['bounding_box']
                
                # Normalize the bounding box coordinates
                width, height = img.size
                x_min, y_min, x_max, y_max = bbox
                
                # Adjust if the model returns normalized coordinates (0-1000)
                # Comment out these lines if the model returns pixel coordinates directly
                x_min = int(x_min / 1000 * width)
                y_min = int(y_min / 1000 * height)
                x_max = int(x_max / 1000 * width)
                y_max = int(y_max / 1000 * height)
                
                norm_bbox = [x_min, y_min, x_max, y_max]
                # print({"label": detection['label'].strip(), "bounding_box": norm_bbox})
                detections.append({"label": detection['label'].strip(), "bounding_box": norm_bbox})
            
            return detections
            
        except Exception as e:
            print(f"Error during Gemma3 inference: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def numpy_to_pil(self, img_np):
        """Converts a NumPy array to a PIL Image."""
        if img_np.ndim == 2:  # Handle grayscale images
            img_pil = Image.fromarray(img_np, 'L')
        elif img_np.ndim == 3:  # Handle color images
            # OpenCV uses BGR format, so we need to convert to RGB
            if img_np.shape[2] == 3:  # Color image
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb, 'RGB')
            elif img_np.shape[2] == 4:  # Color image with alpha
                img_rgba = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGBA)
                img_pil = Image.fromarray(img_rgba, 'RGBA')
            else:
                raise ValueError(f"Unsupported number of channels: {img_np.shape[2]}")
        else:
            raise ValueError(f"Unsupported image ndim: {img_np.ndim}")
        return img_pil
    
    def run_gemma3_detect(self, image_pil, query, visualize=False):
        """
        Run object detection using Gemma3 and optionally visualize the results.
        """
        try:
            detections = self.detect_objects(image_pil, query)
            
            if detections and visualize:
                if isinstance(image_pil, Image.Image):
                    # Ensure the PIL image is in RGB mode
                    if image_pil.mode != "RGB":
                        image_rgb = image_pil.convert("RGB")
                    else:
                        image_rgb = image_pil
                    
                    visualization_image_pil = image_rgb.copy()
                    draw = ImageDraw.Draw(visualization_image_pil)
                    
                    for detection in detections:
                        bbox = detection['bounding_box']
                        x_min, y_min, x_max, y_max = bbox
                        
                        # Draw rectangle
                        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="lime", width=3)
                        
                        label = detection['label']
                        label_text = f"{label}"
                        text_y = max(y_min - 25, 5)
                        text_width = len(label_text) * 8
                        draw.rectangle([(x_min, text_y), (x_min + text_width, text_y + 20)], fill="lime")
                        draw.text((x_min + 5, text_y + 2), label_text, fill="black")
                    
                    # Convert the RGB PIL image to a NumPy array for display
                    visualization_image_np = np.array(visualization_image_pil)
                    # Convert RGB to BGR for OpenCV
                    visualization_image_np = cv2.cvtColor(visualization_image_np, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Detected Objects", visualization_image_np)
                    cv2.waitKey(1)
                else:
                    print("Warning: Input 'image_pil' is not a PIL Image object.")
            
            if detections:
                return detections
            else:
                print(f"No matching objects found or detection failed.")
                return []
                
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            return []

# Example usage
if __name__ == "__main__":
    # Initialize Gemma3 model for object detection
    gemma3_vision = Gemma3Bbox()
    
    # Initialize the camera
    camera = RealSenseCamera(width=640, height=480)
    
    # Main loop
    while True:
        try:
            # Capture image from camera
            img_tmp, _ = camera.get_images()
            img = gemma3_vision.numpy_to_pil(img_tmp)
            
            # Define the object to detect
            object_to_find = "bunny"
            
            # Create detection prompt
            detection_prompt = f"""
            Analyze the image and identify all instances of '{object_to_find}'.
            For each '{object_to_find}' found, output its bounding box coordinates in the format:
            {object_to_find} : (x_min, y_min, x_max, y_max)
            Here, (x_min, y_min) represents the top-left corner of the bounding box, and (x_max, y_max) represents the bottom-right corner.
            If multiple '{object_to_find}' are detected, list each one on a separate line in this exact format, without any additional text or bullet points.
            """
            
            # Alternative prompt for highest confidence detection only
            highc_detection_prompt = f"""
            Analyze the image and identify any instances of '{object_to_find}'.
            From these, determine which '{object_to_find}' instance you are most confident about (the clearest, most certain detection).
            Output **only** the bounding box coordinates of that single most confident '{object_to_find}' in the format:
            {object_to_find} : (x_min, y_min, x_max, y_max)
            (Note: (x_min, y_min) is the top-left corner and (x_max, y_max) is the bottom-right corner of the bounding box.)
            If no '{object_to_find}' is detected in the image, clearly output "No {object_to_find} found."
            """
            
            # Run detection and visualization
            gemma3_vision.run_gemma3_detect(img, highc_detection_prompt, visualize=True)
            
            # Small delay to avoid overwhelming CPU/GPU
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)  # Add delay on error to avoid rapid error loops