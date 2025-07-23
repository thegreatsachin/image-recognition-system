from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model once
model = MobileNetV2(weights='imagenet')

def predict_image_class(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]
    
    return [(label, prob * 100) for (_, label, prob) in decoded]
