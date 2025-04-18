import os
import re
import google.generativeai as genai
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

class GeminiBbox:
    def __init__(self, model_name='gemini-2.0-flash'):
        """
        Initializes the GeminiBbox class using a generative Gemini model.

        Args:
            project_id (str): Your Google Cloud Project ID.
            location (str): The location of the Vertex AI endpoint.
            model_name (str): The name of the generative Gemini model to use (e.g., "gemini-pro-vision").
        """
        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        # Initialize the generative mode
        genai.configure(api_key=api_key)

        # Load the Gemini Pro Vision model
        self.model = genai.GenerativeModel(model_name)
        # Regular expression to find labels before ':' and coordinates inside '()'
        self.bbox_pattern = r"(.+?):\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)"

        # turn on interactive mode
        plt.ion()
        # create a persistent figure and axes
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
                    

    def parse_response(self, text_response):
        # Find all matches in the extracted text
        matches = re.findall(self.bbox_pattern, text_response)
        # print(matches)
        # Process the matches
        detections = []
        for label, x_min, y_min, x_max, y_max in matches:
            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
            detections.append({"label": label.strip(), "bounding_box": bbox})
            # print(f"# Label: {label.strip()}")
            # print(f"# Bounding Box (x_min, y_min, x_max, y_max): {bbox}")
            # print("-" * 20)

        # # The 'detections' list now contains dictionaries with 'label' and 'bounding_box'
        # print("\nExtracted Detections:")
        # for detection in detections:
        #     print(detection)
        
        return detections
        
    def detect_objects(self, img, query):
        try:
            # print(f"Image type: {type(img)}, Shape: {img.size if isinstance(img, Image.Image) else 'N/A'}")
            response = self.model.generate_content([query, img])
            response.resolve()
            temp_detect = self.parse_response(response.text) # sendng label and bbox
            
            detections = []
            for detection in temp_detect:
                # print(detection)
                bbox = detection['bounding_box'] # get the first bbox
                # print(f"Unnormalized Detected bounding box: {bbox}")
                # Normalize the bounding box coordinates
                width, height = img.size
                ymin, xmin, ymax, xmax = bbox
                x_min = int(xmin / 1000 * width)
                y_min = int(ymin / 1000 * height)
                x_max = int(xmax / 1000 * width)
                y_max = int(ymax / 1000 * height)
                norm_bbox = [x_min, y_min, x_max, y_max]
                # print(f"Detected bounding box: {norm_bbox}")
                detections.append({"label": detection['label'].strip(), "bounding_box": norm_bbox})
            
            return detections
        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            return None
    
    
    def run_gemini_detect(self, image_pil, query, visualize=False):
        try:
            detections = self.detect_objects(image_pil, query)

            if detections and visualize:
                if isinstance(image_pil, Image.Image):
                    # print(f"Image format: PIL (mode: {image_pil.mode})")

                    # Ensure the PIL image is in RGB mode
                    if image_pil.mode != "RGB":
                        image_rgb = image_pil.convert("RGB")
                        # print(f"Converted PIL image mode to: {image_rgb.mode}")
                    else:
                        image_rgb = image_pil

                    visualization_image_pil = image_rgb.copy()
                    draw = ImageDraw.Draw(visualization_image_pil)

                    for detection in detections:
                        bbox = detection['bounding_box']
                        # Normalize the bounding box coordinates.
                        width, height = image_rgb.size
                        ymin, xmin, ymax, xmax = bbox
                        x_min = int(xmin / 1000 * width)
                        y_min = int(ymin / 1000 * height)
                        x_max = int(xmax / 1000 * width)
                        y_max = int(ymax / 1000 * height)

                        # print(f"Detection coordinates: (x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max})")
                        
                        # Draw rectangle
                        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="lime", width=3)

                        label = detection['label']
                        label_text = f"{label}"
                        text_y = max(y_min - 25, 5)
                        text_width = len(label_text) * 8
                        draw.rectangle([(x_min, text_y), (x_min + text_width, text_y + 20)], fill="lime")
                        draw.text((x_min + 5, text_y + 2), label_text, fill="black")

                    # Convert the RGB PIL image to a NumPy array for Matplotlib
                    visualization_image_np = np.array(visualization_image_pil)
                    # print(f"Image shape (NumPy for Matplotlib): {visualization_image_np.shape}")

                    # update the existing axes
                    self.ax.clear()
                    self.ax.imshow(visualization_image_np)
                    self.ax.set_title("Detected Objects")
                    self.ax.axis('off')

                    # redraw without blocking
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    # tiny pause to allow GUI event loop to process
                    plt.pause(0.001)

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
    
# Example usage (replace with your actual project ID and image path):
if __name__ == "__main__":
    gemini_vision = GeminiBbox()

    # Example image path (replace with a real image file)
    image_path = "/home/l5vel/NBMOD/a_bunch_of_bananas/img/000006r.png"
    try:
        img = Image.open(image_path)
        
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
        
        gemini_vision.run_gemini_detect(img, highc_detection_prompt)

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")