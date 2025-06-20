from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

MODEL_PATH = 'pneumonia_detection_transfer_learning.h5'
model = load_model(MODEL_PATH)

img_width, img_height = 224, 224

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print("Pneumonia detected")
    else:
        print("Normal")

# Example usage
predict_image(r'E:\pneumonia detection using CNN\data\val\normal\NORMAL2-IM-1427-0001.jpeg')