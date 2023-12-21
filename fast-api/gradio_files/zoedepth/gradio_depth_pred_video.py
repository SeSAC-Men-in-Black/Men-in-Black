import gradio as gr
from mde.ZoeDepth.zoedepth.utils.misc import colorize
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile

# Load YOLO model for vehicle detection
detection_model = YOLO('yolov8m.pt')

vehicles = [1, 2, 3, 5, 7]  # Coco dataset class nums for vehicles
traffic_light = [9]

def cv2_to_pil(cv2_image):
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)
    return pil_image

def get_depth_estimation(frame, depth_model):
    raw_depth = depth_model.infer_pil(frame, pad_input=False, output_type="numpy")
    colored_depth = colorize(raw_depth)
    return raw_depth, colored_depth

def create_demo(depth_model):
    with gr.Blocks() as demo:
        gr.Markdown("### Video Object Detection and Depth Estimation")
        input_video = gr.Video(label="Input Video")

        with gr.Row():
            output_video_1 = gr.Video(label="Video with Detections")
            output_video_2 = gr.Video(label="Depth Map Video")
        with gr.Row():
            output_video_3 = gr.Video(label="Depth Map with Detections and Distance")
            output_video_4 = gr.Video(label="Overlay Video")

        def process_video(video, depth_model):
            try:
                cap = cv2.VideoCapture(video)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
                output_files = [tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) for _ in range(4)]
                video_writers = [cv2.VideoWriter(
                    filename=output_file.name,
                    fourcc=cv2.VideoWriter_fourcc(*'avci'),
                    fps=fps,
                    frameSize=(width, height)
                ) for output_file in output_files]
            
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
            
                    detections = detection_model(frame)[0]
                    frame_with_detections = frame.copy()
                    raw_depth, depth_image = get_depth_estimation(cv2_to_pil(frame), depth_model)
                    depth_image_with_detections = depth_image.copy()
            
                    # Blending the original frame and the depth map for overlay video
                    depth_colored_cv = cv2.cvtColor(np.array(depth_image), cv2.COLOR_RGB2BGR)
                    blended_frame = cv2.addWeighted(frame, 0.6, depth_colored_cv, 0.4, 0)
            
                    for detection in detections.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = detection
                        if int(class_id) in vehicles or int(class_id) in traffic_light:
                            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            depth_value = raw_depth[center_y, center_x]
                            if isinstance(depth_value, np.ndarray):
                                depth_value = np.mean(depth_value)
                            
                            cv2.rectangle(frame_with_detections, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.rectangle(depth_image_with_detections, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(depth_image_with_detections, f"{depth_value:.2f}m", (int(x2), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
                            # Draw rectangles and depth text on the blended frame for overlay video
                            cv2.rectangle(blended_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(blended_frame, f"{depth_value:.2f}m", (int(x2), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
                    # Write the processed frames to the respective video writers
                    video_writers[0].write(frame_with_detections)  # Frame with detections
                    video_writers[1].write(cv2.cvtColor(np.array(depth_image), cv2.COLOR_RGB2BGR))  # Depth map video
                    video_writers[2].write(cv2.cvtColor(np.array(depth_image_with_detections), cv2.COLOR_RGB2BGR))  # Depth map with detections and distance
                    video_writers[3].write(blended_frame)  # Overlay video
            
                # Release all resources
                cap.release()
                for writer in video_writers:
                    writer.release()
            
                output_paths = [output_file.name for output_file in output_files]
                print("Processing complete. Output paths:", output_paths)
                return output_paths
    
            except Exception as e:
                print("Error during video processing:", e)
                return []


        submit = gr.Button("Submit")

        # Add examples
        examples = gr.Examples(
            examples=["examples/test_input_video(5sec)(1920x1080_30FPS).mp4"],
            inputs=[input_video]
        )
        submit.click(
            fn=lambda video: process_video(video, depth_model),
            inputs=[input_video],
            outputs=[output_video_1, output_video_2, output_video_3, output_video_4]
        )