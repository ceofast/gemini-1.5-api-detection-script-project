import os
import json
import numpy as np
import supervision as sv
from roboflow import Roboflow
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_PRIVATE_API_KEY = os.getenv("ROBOFLOW_PRIVATE_API_KEY")


genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")


def validate_google_api_key():
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Test")
        print("Google API Key is valid.")
        return True
    except Exception as e:
        print(f"Error with Google API Key: {str(e)}")
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


def parse_response(response, resolution_wh, classes):
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

check_api_keys()