import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from zoedepth.utils.misc import colorize
import torch.nn as nn
get_ipython().run_line_magic('matplotlib', 'inline')



# Load YOLO model for vehicle detection
coco_model = YOLO('yolov8m.pt')
vehicles = [1, 2, 3, 5, 7]  # Coco dataset class nums for vehicles
traffic_light = [9]


# Load ZoeDepth model for depth estimation
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo

repo = "isl-org/ZoeDepth"
# Zoe_N
# model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

# Zoe_K
# model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True)

# Zoe_NK
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
zoe = model_zoe_nk.to('cuda')

# Function to convert OpenCV image to PIL image
def cv2_to_pil(cv2_image):
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)
    return pil_image

def get_depth_estimation(frame):
    # Get raw depth data (in meters)
    raw_depth = zoe.infer_pil(frame, pad_input=False, output_type="numpy")

    # Colorize for visualization
    colored_depth = colorize(raw_depth)
    return raw_depth, colored_depth


# # Image
image_list = [
    # "/content/Men-in-Black/sample/SeSAC_with_distance_5.83m.jpeg",
    # "/content/Men-in-Black/sample/SeSAC_with_distance_5.88.jpeg",
    # "/content/Men-in-Black/sample/SeSAC_with_distance_5.94m.jpeg",
    "/content/Men-in-Black/sample/SeSAC_without_distance_6m.jpeg",
    "/content/Men-in-Black/sample/SeSAC_without_distance_on_car_3.5m.jpeg",
    "/content/Men-in-Black/sample/SeSAC_without_distance_on_car_6.99m.jpeg"
]

for image_path in image_list:
    frame = cv2.imread(image_path)
    
    if frame is None:
        continue

    detections = coco_model(frame)[0]
    frame_with_detections = frame.copy()
    raw_depth, depth_image = get_depth_estimation(cv2_to_pil(frame))
    depth_image_with_detections = depth_image.copy()

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles or int(class_id) in traffic_light:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            depth_value = raw_depth[center_y, center_x]
            if isinstance(depth_value, np.ndarray):
                depth_value = np.mean(depth_value)
            cv2.rectangle(frame_with_detections, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.rectangle(depth_image_with_detections, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(depth_image_with_detections, f"{depth_value:.2f}m", (int(x2), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_with_detections_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(30, 15))
    for i, (img, title) in enumerate(zip([frame_rgb, frame_with_detections_rgb, depth_image_with_detections], ["Original Frame", "Frame with Detections", "Depth Map with Detections+Distance"])):
        plt.subplot(1, 3, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()


# # Video
# Load video
# video_path = '/content/Men-in-Black/sample_videos/test_input_video(5sec)(1920x1080_30FPS).mp4'
video_path = '/content/Men-in-Black/sample/SeSAC_forward_6.99m_to_1.46m.mp4'
cap = cv2.VideoCapture(video_path)


cap.isOpened()


count = 0
while count<20:
    count+=1
    ret, frame = cap.read()
    if not ret:
        break

    # # Detect objects in the frame
    # detections = coco_model(frame)[0]
    # frame_with_detections = frame.copy()
    # depth_image = get_depth_estimation(cv2_to_pil(frame))

    # Detect objects in the frame
    detections = coco_model(frame)[0]
    frame_with_detections = frame.copy()
    raw_depth, depth_image = get_depth_estimation(cv2_to_pil(frame))
    depth_image_with_detections = depth_image.copy()


    # Process each detection
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles or int(class_id) in traffic_light:
            # Calculate center of the bounding box (you can use this if needed)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
    
            # Get depth value at the center from raw depth data
            depth_value = raw_depth[center_y, center_x]
    
            # Ensure depth_value is a single float value
            if isinstance(depth_value, np.ndarray):
                depth_value = np.mean(depth_value)
    
            # Draw bounding boxes on both images
            cv2.rectangle(frame_with_detections, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.rectangle(depth_image_with_detections, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
            # Text position (top right corner of the bounding box)
            text_position = (int(x2), int(y1))
    
            # Increase font size
            font_scale = 1
    
            # Write depth value on the depth image
            cv2.putText(depth_image_with_detections, f"{depth_value:.2f}m", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    # Convert frames for displaying
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_with_detections_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)

    # Display the images using Matplotlib
    plt.figure(figsize=(30, 15))
    for i, (img, title) in enumerate(zip([frame_rgb, frame_with_detections_rgb, depth_image_with_detections], 
                                         ["Original Frame", "Frame with Detections", "Depth Map with Detections+Distance"])):
        plt.subplot(1, 3, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




