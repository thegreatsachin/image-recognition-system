from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load and preprocess the image
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize for MobileNetV2
    return img_array

# Predict function
def predict_image(img_path):
    processed_image = prepare_image(img_path)
    predictions = model.predict(processed_image)
    decoded = decode_predictions(predictions, top=3)[0]  # Top 3 predictions
    for label in decoded:
        print(f"{label[1]}: {label[2]*100:.2f}%")

# Test with a sample image
if __name__ == "__main__":
    img_path = input("Enter path to the image file: ").strip()
    try:
        predict_image(img_path)
    except Exception as e:
        print("Error:", e)