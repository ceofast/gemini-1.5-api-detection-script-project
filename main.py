import os
import json
import base64
import tempfile
import PIL.Image
import numpy as np
from io import BytesIO
from typing import List
import supervision as sv
from roboflow import Roboflow
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Request


# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_PRIVATE_API_KEY = os.getenv("ROBOFLOW_PRIVATE_API_KEY")


# Ensure the "static" directory exists
os.makedirs("static", exist_ok=True)


def validate_google_api_key():
    """
    Validates the Google Generative AI API key by making a test request.

    Returns:
        bool: True if the API key is valid, False otherwise.

    Prints:
        Confirmation of a valid API key or an error message if the key is invalid.
    """
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
    """
    Validates the Roboflow API key by accessing the workspace.

    Returns:
        bool: True if the API key is valid, False otherwise.

    Prints:
        Confirmation of a valid API key or an error message if the key is invalid.
    """
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        rf.workspace()
        print("Roboflow API key is valid.")
        return True
    except Exception as e:
        print(f"Error with Roboflow API key: {str(e)}")
        return False


def check_api_keys():
    """
    Checks both Google and Roboflow API keys by validating them.

    Raises:
        ValueError: If either the Google or Roboflow API key is invalid.

    Prints:
        A message indicating that API key validation has failed.
    """
    google_valid = validate_google_api_key()
    roboflow_valid = validate_roboflow_api_key()

    if not google_valid or not roboflow_valid:
        raise ValueError("API key validation failed. Please check your .env file.")


# Validate API keys on script initialization
check_api_keys()


# Configure the Google Generative AI model
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")


# Initialize FastAPI app
app = FastAPI()


# Mount the "static" directory for serving static files like images and CSS
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class DetectionResponse(BaseModel):
    """
    Model for the response object containing detections and the annotated image.

    Attributes:
        detections (List[dict]): A list of detected objects with their bounding box coordinates.
        annotated_image (str): Base64 encoded string of the annotated image.
    """
    detections: List[dict]
    annotated_image: str


def parse_response(response, resolution_wh, classes: List[str]):
    """
    Parses the response from the generative AI model to extract object detection data.

    Args:
        response (Response): The response object from the AI model.
        resolution_wh (tuple): A tuple containing the width and height of the image.
        classes (list): A list of class names to detect in the image.

    Returns:
        sv.Detections: A Detections object containing bounding boxes and class information for detected objects.
    """
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


# Template for generating prompts for object detection
PROMPT_TEMPLATE = 'Return bounding boxes for `{}` as JSON arrays [ymin, xmin, ymax, xmax]. For example ```json\n[\n  {{\n    "person": [\n      255,\n      69,\n      735,\n      738\n    ]\n  }}\n]\n```'


@app.post("/detect_objects/", response_class=FileResponse)
async def detect_objects(request: Request, image: UploadFile = File(...), classes: str = Form(...)):
    """
    Endpoint to detect objects in an uploaded image.

    Args:
        request (Request): The FastAPI request object.
        image (UploadFile): The image file uploaded by the user.
        classes (str): A comma-separated string of class names to detect in the image.

    Returns:
        FileResponse: A PNG image file with annotated detections.

    Raises:
        HTTPException: If an error occurs during processing, a 500 error is returned with the error details.
    """
    try:
        # Parse the list of classes from the provided comma-separated string
        class_list = [c.strip() for c in classes.split(",")]

        # Format the prompt for object detection based on provided classes
        prompt = PROMPT_TEMPLATE.format(", ".join(class_list))

        # Read the image content and convert it to a PIL Image
        content = await image.read()
        pil_image = PIL.Image.open(BytesIO(content))
        np_image = np.array(pil_image)

        # Generate object detection content using the AI model
        response = model.generate_content([prompt, pil_image])

        # Parse the model's response to extract detections
        detections = parse_response(response, pil_image.size, class_list)

        # Initialize annotators for bounding boxes and labels
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # Annotate the image with bounding boxes and labels
        annotated_image = box_annotator.annotate(scene=np_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # Save the annotated image to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        PIL.Image.fromarray(annotated_image).save(temp_file.name)

        # Return the annotated image as a file response
        return FileResponse(temp_file.name, media_type="image/png")

    except Exception as e:
        # Raise an HTTP 500 error if any exceptions occur
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)