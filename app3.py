from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import os

# Create directories for static files and templates
image_directory = "uploaded_images"
os.makedirs(image_directory, exist_ok=True)
templates_directory = "templates"

# Initialize FastAPI app
app = FastAPI()

# Serve images statically
app.mount("/images", StaticFiles(directory=image_directory), name="images")

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This should be more restrictive in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 template rendering setup
templates = Jinja2Templates(directory=templates_directory)

# Load the TensorFlow model
def load_model():
    model_path =r"C:\Users\Varun S\Downloads\Osteoporosis_model2.h5"  # Adjust this to the path of your TensorFlow model
    return tf.keras.models.load_model(model_path)

model = load_model()

def preprocess_image(img):
    """
    Preprocess the image to fit the input requirements of the TensorFlow model and enhance image clarity:
    - Convert to grayscale
    - Apply Gaussian Blur to reduce noise
    - Use adaptive thresholding to enhance image contrast
    - Resize to the model's expected input size (e.g., 224x224)
    - Normalize pixel values to [0, 1]
    """
    img = img.convert('L')  # Convert image to grayscale
    img_array = np.array(img)  # Convert PIL image to numpy array

    # Apply Gaussian Blur
    img_blurred = cv2.GaussianBlur(img_array, (5, 5), 0)

    # Apply adaptive thresholding to enhance contrast
    img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

    # Resize image
    img_resized = cv2.resize(img_thresh, (224, 224), interpolation=cv2.INTER_AREA)

    # Normalize the image
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Expand dimensions to match model's input
    img_expanded = np.expand_dims(img_normalized, axis=0)  # for batch size
    img_expanded = np.expand_dims(img_expanded, axis=-1)   # for channel

    return img_expanded

def predict(model, preprocessed_image):
    """ Use the loaded model to predict if the image indicates osteoporosis. """
    prediction = model.predict(preprocessed_image)
    return "Osteoporosis" if prediction[0][0] > 0.5 else "Not Osteoporosis"

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """ Serve the main page. """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    """ Handles image uploads and displays prediction results. """
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    image_path = os.path.join(image_directory, file.filename)
    image.save(image_path)  # Save image to static directory

    preprocessed_image = preprocess_image(image)
    prediction_result = predict(model, preprocessed_image)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "filename": file.filename,
        "prediction": prediction_result,
        "image_url": f"/images/{file.filename}"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
