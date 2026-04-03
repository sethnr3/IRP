import base64
import numpy as np
import cv2
import random

def predict_face_emotion(image_base64):
    try:
        img_data = base64.b64decode(image_base64.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # TEMP (replace with VGG19 FER model)
        emotions = ["sadness", "joy", "neutral"]
        emotion = random.choice(emotions)
        confidence = round(random.uniform(0.6, 0.9), 2)

        return emotion, confidence

    except:
        return "No face detected", 0.0