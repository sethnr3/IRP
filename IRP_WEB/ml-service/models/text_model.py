import random

def predict_text_emotion(text):
    # TEMP (replace with your trained model)
    emotions = ["sadness", "joy", "anger", "fear", "neutral"]
    emotion = random.choice(emotions)
    confidence = round(random.uniform(0.7, 0.95), 2)
    return emotion, confidence