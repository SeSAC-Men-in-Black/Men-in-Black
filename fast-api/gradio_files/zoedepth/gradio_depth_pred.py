import gradio as gr
from mde.ZoeDepth.zoedepth.utils.misc import colorize
from PIL import Image
import tempfile
import numpy as np
import cv2
from ultralytics import YOLO


# Load YOLO model for vehicle detection
detection_model = YOLO('yolov8m.pt')

vehicles = [1, 2, 3, 5, 7]  # Coco dataset class nums for vehicles
traffic_light = [9]

# Function to convert OpenCV image to PIL image
def cv2_to_pil(cv2_image):
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)
    return pil_image

def get_depth_estimation(frame, depth_model):
    # Get raw depth data (in meters)
    # Assuming depth_model.infer_pil expects a PIL image and returns a numpy array
    raw_depth = depth_model.infer_pil(frame, pad_input=False, output_type="numpy")

    # Colorize for visualization
    colored_depth = colorize(raw_depth)

    return raw_depth, colored_depth


def create_demo(depth_model):
    with gr.Blocks() as demo:
        gr.Markdown("### Object Detection(Yolo) and Depth Estimation(ZoeDepth)")
        input_image = gr.Image(label="Input Image", type='pil')
        with gr.Row():
            
            output_image_1 = gr.Image(label="Frame with Detections")
            output_image_2 = gr.Image(label="Depth Map")
            output_image_3 = gr.Image(label="Depth Map with Detections and Distance")

        def process_image(image):
            # Convert PIL image to OpenCV format
            open_cv_image = np.array(image)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

            # Object detection
            detections = detection_model(open_cv_image)[0]
            frame_with_detections = open_cv_image.copy()

            # Depth estimation
            pil_image = cv2_to_pil(open_cv_image)
            raw_depth, colored_depth = get_depth_estimation(pil_image, depth_model)
            depth_image_with_detections = colored_depth.copy()

            # Draw detections and depth values
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles or int(class_id) in traffic_light:
                    # Calculate the center of the detection
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
            
                    # Get the depth value at the center of the detection
                    depth_value = raw_depth[center_y, center_x]
                    if isinstance(depth_value, np.ndarray):
                        depth_value = np.mean(depth_value)
            
                    # Draw a rectangle around the detection on the original frame
                    cv2.rectangle(frame_with_detections, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw a rectangle around the detection on the depth map
                    cv2.rectangle(depth_image_with_detections, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Put the depth value text on the depth map
                    cv2.putText(depth_image_with_detections, f"{depth_value:.2f}m", (int(x2), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            # Convert back to PIL for Gradio output
            frame_with_detections_pil = cv2_to_pil(frame_with_detections)
            depth_image_with_detections_pil = cv2_to_pil(depth_image_with_detections)

            return frame_with_detections_pil, colored_depth, depth_image_with_detections_pil

        # Add examples
        examples = gr.Examples(
            examples=["examples/SeSAC_without_distance_on_car_3.5m.jpeg"],
            inputs=[input_image]
        )

        submit = gr.Button("Submit")
        submit.click(process_image, inputs=[input_image], outputs=[output_image_1, output_image_2, output_image_3])