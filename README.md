# gemini-1.5-api-detection-script-project

This FastAPI application allows you to upload an image and detect objects within it using Google's Gemini model and Roboflow's API. The application returns the detected objects' bounding boxes directly as an annotated image.


## Features
- **Object Detection**: Detect objects in an image using predefined classes.
- **Image Annotation**: Annotate the detected object directly on the image.
- **API Key Validation**: Validate Google and Roboflow API keys before running the application.
- **Dynamic Image Processing**: Handle and process images dynamically, responding with the annotated image.


## Requirements
- Python 3.11+
- FastAPI
- Google Generative AI (Gemini)
- Roboflow API
- Supervision Library
- PIL (Pillow)
- dotenv


## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ceofast/gemini-1.5-api-detection-script-project.git
```

```cd fastapi-object-detection```

### 2. Set Up a Virtual Environment
* On macOS and Linux
```bash
python3 -m venv env
source env/bin/activate
```

On Windows:
```bash
.\env\Scripts\activate
```

Using `conda`
1. **Create a conda environment**:
```bash
conda create --name myenv python==3.11
```

2. **Activate the conda environment**:
```bash
conda activate myenv
```


### 3. Install Dependencies
After setting up the virtual environment, install the required dependencies:

```bash
pip install -r requirements.txt
```


### Set Up Environment Variables
1. **Create a `.env` file** in the root directory.
2. **Add your API keys to the `.env` file**:

```bash
GOOGLE_API_KEY=your_google_api_key
ROBOFLOW_API_KEY=your_roboflow_api_key
```

### Running FastAPI application:
1. **Run the FastAPI application**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

2. **Access the application**: Open your web browser and navigate to `http://localhost:8000/docs` to view the Swagger UI documentation for the API.

3. **Upload an image and detect objects**:

* Use the `/detect_objects/` endpoint to upload an image and specify the classes you want to detect.
* The response will be the annotated image showing the detected objects.


## Docker Setup

### Dockerfile
To containerize the FastAPI application, you can use the following Dockerfile

#### Building and Running the Docker Container
1. **Build the Docker image**:
```bash
docker build -t my-fastapi-app .
```

2. **Run the Docker container**:
```bash
docker run -d -p 8000:8000 my-fastapi-app
```

3. **Access the application**:
* After running the container, the application will be accessible at `htttp://localhost:8000`

## API Endpoints
**POST`/detect_objects/`**

* **Parameters**:
  * `ìmage`: The image file to be processed.
  * `classes`: A comma-seperated list of classes to detect (e.g., "person, car, dog").
* **Response**:
  * Returns the annotated image with bounding boxes around the detected objects.


## Example
Here is how to use the API:

1. Upload an image and specify the classes:
```bash
curl -X POST "http://localhost:8000/detect_objects/" -F "image=@path_to_your_image.png" -F "classes=person, dog, car"
```
2. The API will respond with the image annotated with bounding boxes around the detected objects.

## Environment Variables
* **GOOGLE_API_KEY**: Your Google API key for accessing the Gemini model.
* **ROBOFLOW_API_KEY**: Your Roboflow API key for object detection.
* **Optional**: Other environment variables can be added as needed.


## Contributing
If you would like to contribute to this project, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

