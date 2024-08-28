import tempfile
import numpy as np
from PIL import Image
from io import BytesIO
import supervision as sv
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from models import model, parse_response, PROMPT_TEMPLATE
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request


app = FastAPI()

# Mount the "static" directory for serving static files like images and CSS
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


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
        # Parse the list of classes from the provided comma-seperated string
        class_list = [c.strip() for c in classes.split(",")]
        # Format the prompt for object detection based on provided classes
        prompt = PROMPT_TEMPLATE.format(", ".join(class_list))

        # Read the image content and convert it to a PIL Image
        content = await image.read()
        pil_image = Image.open(BytesIO(content))
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
        Image.fromarray(annotated_image).save(temp_file.name)

        # Return the annotated image as a file response
        return FileResponse(temp_file.name, media_type="image/png")

    except Exception as e:
        # Raise an HTTP 500 error if any exceptions occur
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)