import os
import json
import base64
from io import BytesIO
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import google.generativeai as genai
import numpy as np
import PIL.Image
import supervision as sv
import tempfile
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_PRIVATE_API_KEY = os.getenv("ROBOFLOW_PRIVATE_API_KEY")

os.makedirs("static", exist_ok=True)


def validate_google_api_key():
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Test")
        print("Google API key is valid.")
        return True
    except Exception as e:
        print(f"Error with Google API key: {str(e)}")
        return False


def validate_roboflow_api_key():
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        rf.workspace()
        print("Roboflow API key is valid.")
        return True
    except Exception as e:
        print(f"Error with Roboflow API key: {str(e)}")
        return False


def check_api_keys():
    google_valid = validate_google_api_key()
    roboflow_valid = validate_roboflow_api_key()

    if not google_valid or not roboflow_valid:
        raise ValueError("API key validation failed. Please check your .env file.")


check_api_keys()

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class DetectionResponse(BaseModel):
    detections: List[dict]
    annotated_image: str


def parse_response(response, resolution_wh, classes: List[str]):
    w, h = resolution_wh
    data = json.loads(response.text.replace('json', '').replace('```', '').replace('\n', ''))
    class_name = []
    for i in range(len(data)):
        class_name += list(data[i].keys())
    yxyx = []
    for i in range(len(data)):
        yxyx += list(data[i].values())
    yxyx = np.array(yxyx, dtype=np.float64)
    xyxy = yxyx[:, [1, 0, 3, 2]]
    xyxy /= 1000
    xyxy *= np.array([w, h, w, h])

    detections = sv.Detections(
        xyxy=xyxy,
        class_id=np.array([classes.index(i) for i in class_name]),
    )
    detections.data["class_name"] = class_name

    return detections


PROMPT_TEMPLATE = 'Return bounding boxes for `{}` as JSON arrays [ymin, xmin, ymax, xmax]. For example ```json\n[\n  {{\n    "person": [\n      255,\n      69,\n      735,\n      738\n    ]\n  }}\n]\n```'


@app.post("/detect_objects/", response_class=FileResponse)
async def detect_objects(
    request: Request,
    image: UploadFile = File(...),
    classes: str = Form(...)
):
    try:
        class_list = [c.strip() for c in classes.split(",")]

        prompt = PROMPT_TEMPLATE.format(", ".join(class_list))

        content = await image.read()
        pil_image = PIL.Image.open(BytesIO(content))
        np_image = np.array(pil_image)

        response = model.generate_content([prompt, pil_image])

        detections = parse_response(response, pil_image.size, class_list)

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = box_annotator.annotate(scene=np_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        PIL.Image.fromarray(annotated_image).save(temp_file.name)

        return FileResponse(temp_file.name, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)