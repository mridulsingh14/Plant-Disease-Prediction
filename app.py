# Remove Gemini/Generative AI imports and code
# import google.generativeai as genai  # REMOVE THIS LINE
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.models import load_model
import json
import requests
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.applications.inception_v3 import preprocess_input as inception_preprocess
import shutil
import random
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Load environment variables from a .env file
load_dotenv()

# Set up the model configuration for text generation
# generation_config = {...}  # REMOVE

# Define safety settings for content generation
# safety_settings = [...]  # REMOVE

# Initialize the GenerativeModel with the specified model name, configuration, and safety settings
# model = genai.GenerativeModel(...)  # REMOVE

# Function to read image data from a file path
def read_image_data(file_path):
    image_path = Path(file_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find image: {image_path}")
    return {"mime_type": "image/jpeg", "data": image_path.read_bytes()}

# Replace Gemini model with a generic model interface

def is_leaf_label(label):
    # Expanded keywords for better leaf/plant detection
    leaf_keywords = [
        'leaf', 'leaves', 'foliage', 'plant', 'tree', 'herb', 'shrub', 'flora',
        'vegetation', 'branch', 'stem', 'vine', 'sapling', 'seedling',
        # Add common plant/leaf names from ImageNet
        'maple', 'oak', 'ash', 'willow', 'poplar', 'birch', 'elm', 'sycamore',
        'beech', 'chestnut', 'linden', 'mulberry', 'fig', 'cypress', 'acacia',
        'eucalyptus', 'bamboo', 'banana', 'cabbage', 'lettuce', 'spinach',
        'kale', 'mint', 'basil', 'parsley', 'cilantro', 'thyme', 'rosemary',
        'sage', 'dill', 'oregano', 'lavender', 'ivy', 'clover', 'fern', 'moss'
    ]
    return any(keyword in label.lower() for keyword in leaf_keywords)

# Initial input prompt for the plant pathologist
input_prompt = """
As a highly skilled plant pathologist, your expertise is indispensable in our pursuit of maintaining optimal plant health. You will be provided with information or samples related to plant diseases, and your role involves conducting a detailed analysis to identify the specific issues, propose solutions, and offer recommendations.

**Analysis Guidelines:**

1. **Disease Identification:** Examine the provided information or samples to identify and characterize plant diseases accurately.

2. **Detailed Findings:** Provide in-depth findings on the nature and extent of the identified plant diseases, including affected plant parts, symptoms, and potential causes.

3. **Next Steps:** Outline the recommended course of action for managing and controlling the identified plant diseases. This may involve treatment options, preventive measures, or further investigations.

4. **Recommendations:** Offer informed recommendations for maintaining plant health, preventing disease spread, and optimizing overall plant well-being.

5. **Important Note:** As a plant pathologist, your insights are vital for informed decision-making in agriculture and plant management. Your response should be thorough, concise, and focused on plant health.

**Disclaimer:**
*"Please note that the information provided is based on plant pathology analysis and should not replace professional agricultural advice. Consult with qualified agricultural experts before implementing any strategies or treatments."*

Your role is pivotal in ensuring the health and productivity of plants. Proceed to analyze the provided information or samples, adhering to the structured 
"""

# Load class indices from file (generated during training)
indices_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_indices.json')
with open(indices_path, 'r') as f:
    class_indices = json.load(f)
print("Class indices mapping:", class_indices)
print("Class order:", [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])])

# Ensure class_names is ordered by index
class_names = [None] * len(class_indices)
for k, v in class_indices.items():
    class_names[v] = k

# Paths to both .h5 models
PLANT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plant_disease_model.h5')
INCEPTION_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'InceptionV3_plant_disease_model.h5')

# Load plant model
plant_model = load_model(PLANT_MODEL_PATH, compile=False)
# Try to load InceptionV3 model
try:
    inception_model = load_model(INCEPTION_MODEL_PATH, compile=False)
    inception_model_loaded = True
except Exception as e:
    inception_model = None
    inception_model_loaded = False
    inception_model_error = str(e)

def predict_with_both_models(file_path):
    # For Custom CNN
    img_cnn = Image.open(file_path).convert('RGB').resize((224, 224))
    x_cnn = np.array(img_cnn) / 255.0
    x_cnn = np.expand_dims(x_cnn, axis=0)
    plant_preds = plant_model.predict(x_cnn)

    # For InceptionV3
    img_incep = Image.open(file_path).convert('RGB').resize((299, 299))
    x_incep = np.array(img_incep)
    x_incep = inception_preprocess(x_incep)
    x_incep = np.expand_dims(x_incep, axis=0)
    inception_preds = inception_model.predict(x_incep)

    plant_class_id = np.argmax(plant_preds)
    plant_confidence = np.max(plant_preds) * 100
    plant_class_name = class_names[plant_class_id] if plant_class_id < len(class_names) else str(plant_class_id)
    if plant_confidence < 70:
        plant_result = "invalid"
    else:
        plant_result = f"PlantVillage Prediction: {plant_class_name}\nConfidence: {plant_confidence:.2f}%"

    inception_class_id = np.argmax(inception_preds)
    inception_confidence = np.max(inception_preds) * 100
    inception_class_name = class_names[inception_class_id] if inception_class_id < len(class_names) else str(inception_class_id)
    if inception_confidence < 70:
        inception_result = "invalid"
    else:
        inception_result = f"InceptionV3 Prediction: {inception_class_name}\nConfidence: {inception_confidence:.2f}%"

    return plant_result, inception_result

def process_uploaded_files(files):
    file_path = files[0].name if files else None
    if file_path:
        plant_result, inception_result = predict_with_both_models(file_path)
        combined_result = f"{plant_result}\n\n---\n\nInceptionV3 Model Prediction:\n{inception_result}"
    else:
        combined_result = None
    return file_path, combined_result

# Gradio interface setup
with gr.Blocks() as demo:
    file_output = gr.Textbox()
    image_output = gr.Image()
    combined_output = [image_output, file_output]

    upload_button = gr.UploadButton(
        "Click to Upload an Image",
        file_types=["image"],
        file_count="multiple",
    )
    upload_button.upload(process_uploaded_files, upload_button, combined_output)

demo.launch(debug=True, share=True)

# For Custom CNN
img_cnn = Image.open('path_to_sample_image.jpg').convert('RGB').resize((224, 224))
x_cnn = np.array(img_cnn) / 255.0
print("Custom CNN input min/max:", x_cnn.min(), x_cnn.max())

# For InceptionV3
img_incep = Image.open('path_to_sample_image.jpg').convert('RGB').resize((299, 299))
x_incep = np.array(img_incep)
x_incep = inception_preprocess(x_incep)
print("InceptionV3 input min/max:", x_incep.min(), x_incep.max())

DATASET_DIR = 'PlantdiseaseDetectionApp/PlantVillage'
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
print("Class distribution in training set:", dict(zip(train_gen.class_indices.keys(), np.bincount(train_gen.classes))))
