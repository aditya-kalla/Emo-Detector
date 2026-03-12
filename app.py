import gradio as gr
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- MODEL AND CLASSIFIER LOADING ---
model_path = "emotion_model.h5"
model = None

# Load model safely
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: Model file not found at '{model_path}'.")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# FIX: You MUST download 'haarcascade_frontalface_default.xml' 
# and upload it to your repository for this line to work.
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def predict_emotion(frame):
    if frame is None or model is None:
        return None, "System Error: Model or Input Missing"
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    label = "No Face Detected"
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            prediction = model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    return frame, label

# --- GRADIO INTERFACE ---
demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy"), gr.Label()],
    title="Emo Detector",
    description="Detect emotions from images."
)

if __name__ == "__main__":
    demo.launch()