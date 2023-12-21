from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.templating import Jinja2Templates
import torch
import gradio as gr
from gradio_files.zoedepth.gradio_depth_pred import create_demo as create_depth_pred_demo
from gradio_files.zoedepth.gradio_im_to_3d import create_demo as create_im_to_3d_demo
from gradio_files.zoedepth.gradio_pano_to_3d import create_demo as create_pano_to_3d_demo
from gradio_files.zoedepth.gradio_depth_pred_video import create_demo as create_depth_pred_demo_video
import threading
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7863"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Static files
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Jinja2 Templates
templates = Jinja2Templates(directory="templates")

# Home page
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

    
# ZoeDepth Demo css
css = """
#img-display-container {
    max-height: 50vh;
    }
#img-display-input {
    max-height: 40vh;
    }
#img-display-output {
    max-height: 40vh;
    }
"""

# Load ZoeDepth Model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
depth_model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_NK", pretrained=True).to(DEVICE).eval()

title = "# Calculate Distance to vehicle and traffic light with Zoedepth and Yolo"
description = "test"
#"""Official demo for **ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth**.

# ZoeDepth is a deep learning model for metric depth estimation from a single image.

# Please refer to our [paper](https://arxiv.org/abs/2302.12288) or [github](https://github.com/isl-org/ZoeDepth) for more details."""

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Tab("Image Depth Prediction"):
        create_depth_pred_demo(depth_model)
    with gr.Tab("Video Depth Prediction"):
        create_depth_pred_demo_video(depth_model)
    with gr.Tab("Image to 3D"):
        create_im_to_3d_demo(depth_model)
    with gr.Tab("360 Panorama to 3D"):
        create_pano_to_3d_demo(depth_model)

# Gradio 인터페이스를 FastAPI 앱에 마운트
app = gr.mount_gradio_app(app, demo, path="/zoedepth_demo")


# ZoeDepth Demo Page
@app.get("/zoedepth")
async def get_zoedepth(request: Request):
    # Simply render the template with the iframe
    return templates.TemplateResponse("zoedepth_demo.html", {"request": request})
