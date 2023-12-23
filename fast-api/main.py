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
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Static files
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
# Gradio 파일이 저장되는 디렉토리를 정적 파일 경로로 설정
app.mount("/static", StaticFiles(directory="/tmp/gradio"), name="static")

# Jinja2 Templates
templates = Jinja2Templates(directory="templates")

# Home page
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Paper Review end-to-end-Learning
@app.get("/paper_end-to-end-Learning")
async def paper_end_to_end_Learning(request: Request):
    return templates.TemplateResponse("paper_end-to-end-Learning.html", {"request": request})

# Paper Review MonoDepth2
@app.get("/paper_MonoDepth2")
async def paper_MonoDepth2(request: Request):
    return templates.TemplateResponse("paper_MonoDepth2.html", {"request": request})

# Paper Review VDE
@app.get("/paper_VDE")
async def paper_VDE(request: Request):
    return templates.TemplateResponse("paper_VDE.html", {"request": request})

# Paper Review ZoeDepth
@app.get("/paper_ZoeDepth")
async def paper_ZoeDepth(request: Request):
    return templates.TemplateResponse("paper_ZoeDepth.html", {"request": request})

# Traffic Light Recognition
@app.get("/Traffic_Light_Recognition")
async def traffic_light_recognition(request: Request):
    return templates.TemplateResponse("traffic_light_recognition.html", {"request": request})

# License Plate Recognition
@app.get("/License_Plate_Recognition")
async def license_plate_recognition(request: Request):
    return templates.TemplateResponse("license_plate_recognition.html", {"request": request})

# Line Violation Detection
@app.get("/Line_Violation_Detection")
async def line_violation_detection(request: Request):
    return templates.TemplateResponse("line_violation_detection.html", {"request": request})


# demo_license
@app.get("/demo_license")
async def demo_license(request: Request):
    return templates.TemplateResponse("demo_license.html", {"request": request})

# demo_line
@app.get("/demo_line")
async def demo_line(request: Request):
    return templates.TemplateResponse("demo_line.html", {"request": request})

# demo_traffic
@app.get("/demo_traffic")
async def demo_traffic(request: Request):
    return templates.TemplateResponse("demo_traffic.html", {"request": request})

# demo_monodepth2
@app.get("/demo_monodepth2")
async def demo_monodepth2(request: Request):
    return templates.TemplateResponse("demo_monodepth2.html", {"request": request})


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

title = "# Distance Calculation to vehicle & traffic light using Zoedepth and Yolo"

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
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
