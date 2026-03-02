import gradio as gr
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os

# --- MODEL AND CLASSIFIER LOADING ---
# Define the model path
model_path = "emotion_model.h5"
model = None

# Load the trained emotion detection model
# Use a try-except block to handle potential errors gracefully
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("The application will run, but predictions will be disabled.")
else:
    print(f"Warning: Model file not found at '{model_path}'.")
    print("The application will run, but predictions will be disabled.")


# Emotion labels (in order matching the model's output)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Haarcascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- STYLING AND APPEARANCE ---
# Define colors for each emotion for a more dynamic and intuitive UI
emotion_colors = {
    'Angry': (0, 0, 255),       # Red
    'Disgust': (0, 128, 0),     # Dark Green
    'Fear': (128, 0, 128),      # Purple
    'Happy': (0, 255, 255),     # Yellow
    'Neutral': (255, 255, 255), # White
    'Sad': (255, 0, 0),         # Blue
    'Surprise': (255, 165, 0)   # Orange
}

# Custom CSS for a modern, aesthetic, and user-friendly look
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

body {
    background: linear-gradient(135deg, #1f2937, #111827);
    font-family: 'Poppins', sans-serif;
}
.gradio-container {
    border-radius: 20px !important;
    background: rgba(31, 41, 55, 0.8);
    backdrop-filter: blur(10px);
}
h1, h2, h3, p {
    color: #f3f4f6;
    text-align: center;
}
h1 {
    font-size: 2.5em;
    font-weight: 600;
    letter-spacing: 2px;
}
.gr-button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white !important;
    font-weight: bold;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}
.gr-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}
.tabs button {
    background-color: #374151 !important;
    color: #d1d5db !important;
    border-radius: 8px 8px 0 0 !important;
    border: none !important;
    transition: background-color 0.3s;
}
.tabs button.selected {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
}
label {
    color: #d1d5db !important;
    font-weight: 400;
}
.output_image, .input_image {
    border-radius: 15px !important;
    overflow: hidden;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
"""

# --- CORE PREDICTION FUNCTION (IMPROVED FOR MULTI-FACE DETECTION) ---
def predict_emotion(image):
    """
    Detects faces in an image, predicts the emotion for each face,
    and draws colored bounding boxes and labels on the image.
    """
    # Handle cases where no image is provided or the model failed to load
    if image is None:
        return None, "Please upload an image first! 🖼️"
    if model is None:
        # Return the original image with an error message if the model isn't loaded
        cv2.putText(image, "Model not loaded.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return image, "Model not loaded. Cannot perform prediction."

    # Convert the input image from RGB (Gradio) to BGR (OpenCV)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Convert to grayscale for the face detector
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    
    detections = []
    
    # Loop through all detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (the face)
        roi_gray = gray[y:y + h, x:x + w]
        # Resize to the model's expected input size (48x48)
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Preprocess the image for the model
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Make a prediction
            preds = model.predict(roi, verbose=0)[0]
            confidence = np.max(preds)
            label = emotion_labels[preds.argmax()]
            
            # Store the detection details
            detections.append({
                'box': (x, y, w, h),
                'label': label,
                'confidence': confidence,
                'area': w * h
            })

    # Handle case where no faces are detected
    if not detections:
        cv2.putText(img_bgr, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb, "No Face Detected 😔"

    # Draw bounding boxes and labels on the original image for all detected faces
    for det in detections:
        x, y, w, h = det['box']
        label = det['label']
        confidence = det['confidence']
        color = emotion_colors.get(label, (255, 255, 255)) # Default to white if emotion not in dict
        
        # Draw the bounding box
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
        
        # Create the label text with emotion and confidence percentage
        label_text = f"{label} ({confidence*100:.1f}%)"
        
        # Calculate text size to draw a filled rectangle as a background for the text
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_bgr, (x, y - text_height - 10), (x + text_width, y), color, -1)
        # Put the text on the image. The text color is black for better contrast on the colored background.
        cv2.putText(img_bgr, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Determine the primary emotion based on the largest detected face
    largest_face = max(detections, key=lambda det: det['area'])
    primary_label = f"Primary Emotion: {largest_face['label']}"

    # Convert the image back to RGB for Gradio display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return img_rgb, primary_label


# --- GRADIO UI DEFINITION ---
with gr.Blocks(css=custom_css, theme=gr.themes.Base(primary_hue="blue", secondary_hue="purple")) as demo:
    gr.Markdown("<h1>🌟 Emo-Sense AI 🎥</h1>")
    gr.Markdown("<p>Instantly detect emotions from faces using a deep learning model. <br> Upload an image or use your live webcam feed.</p>")

    with gr.Tabs():
        # --- Tab for Uploading an Image ---
        with gr.TabItem("🖼️ Upload Image"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    upload_img = gr.Image(sources=["upload"], type="numpy", label="Upload Your Image")
                    gr.Markdown("### How it Works:")
                    gr.Markdown("- Upload an image containing one or more faces.\n- The AI will detect all faces and classify their emotions.\n- Bounding box colors correspond to the detected emotion.")
                    submit_btn = gr.Button("Detect Emotions", variant="primary")
                with gr.Column(scale=1):
                    img_output = gr.Image(label="Processed Image", elem_classes="output_image")
                    img_label = gr.Label(label="Prediction Result")
            submit_btn.click(predict_emotion, inputs=upload_img, outputs=[img_output, img_label])

        # --- Tab for Live Webcam Feed ---
        with gr.TabItem("📷 Live Webcam"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    webcam_stream = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Live Webcam Feed")
                with gr.Column(scale=1):
                    live_output = gr.Image(label="Live Detection", elem_classes="output_image")
                    live_label = gr.Label(label="Detected Emotion")
            
            # The streaming function will call our improved predict_emotion function continuously
            webcam_stream.stream(predict_emotion, inputs=webcam_stream, outputs=[live_output, live_label])

    gr.Markdown("<p style='text-align: center; color: #9ca3af; margin-top: 2rem;'>💡 Tip: Ensure good lighting and clear face visibility for best results.</p>")

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(debug=True)
