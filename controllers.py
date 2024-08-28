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

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.post("/detect_objects/", response_class=FileResponse)
async def detect_objects(request: Request, image: UploadFile = File(...), classes: str = Form(...)):
    try:
        class_list = [c.strip() for c in classes.split(",")]
        prompt = PROMPT_TEMPLATE.format(", ".join(class_list))

        content = await image.read()
        pil_image = Image.open(BytesIO(content))  # BytesIO çağrısı için düzeltme yapıldı
        np_image = np.array(pil_image)

        response = model.generate_content([prompt, pil_image])
        detections = parse_response(response, pil_image.size, class_list)

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = box_annotator.annotate(scene=np_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        Image.fromarray(annotated_image).save(temp_file.name)

        return FileResponse(temp_file.name, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)