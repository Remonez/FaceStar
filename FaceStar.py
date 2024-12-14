import os
import streamlit as st
import cv2
import numpy as np
import joblib
import json
import pywt

# Load Haar Cascade for face detection
face_cascade_path = os.path.join(os.path.dirname(__file__), 'opencv', 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(os.path.dirname(__file__), 'opencv', 'haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Function for wavelet transformation
def w2d(img, mode='haar', level=5):
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray) / 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    return np.uint8(imArray_H * 255)

# Load your trained model pipeline
model_pipeline = joblib.load('models/svmsaved_model.pkl')

# Load the class dictionary

class_dict = {"Denzel Washington": 0, "Leonardo DiCaprio": 1, "Robert Downey Jr": 2, "Sandra Bullock": 3, "Scarlett Johansson": 4}

# Additional details for each character
details_dict = {
    'Denzel Washington': "A powerhouse of talent, Denzel Washington is an American actor, director, and producer. Known for his dynamic performances, he has won two Academy Awards and numerous accolades for his roles in films like 'Training Day' and 'Fences.' Beyond acting, Denzel is a prominent advocate for social justice and representation in the arts.",
    'Leonardo DiCaprio': "An emblem of modern cinema, Leonardo DiCaprio is renowned for his versatility and commitment to his roles. With an Oscar for 'The Revenant' and multiple nominations for films like 'The Wolf of Wall Street' and 'Inception,' he continues to captivate audiences. DiCaprio is also a passionate environmentalist, using his platform to promote sustainability.",
    'Robert Downey Jr': "As one of Hollywood's most celebrated actors, Robert Downey Jr. has transformed the superhero genre with his iconic portrayal of Iron Man. With a career spanning decades, he has delivered acclaimed performances in films like 'Chaplin' and 'Sherlock Holmes.' Downey's charisma and charm have made him a beloved figure both on and off the screen.",
    'Sandra Bullock': "A beloved actress and producer, Sandra Bullock has charmed audiences with her roles in romantic comedies like 'While You Were Sleeping' and thrillers like 'Gravity.' With an Academy Award for 'The Blind Side,' she is celebrated for her versatility and philanthropic efforts, making a significant impact in the entertainment industry.",
    'Scarlett Johansson': "Scarlett Johansson is a dynamic force in Hollywood, known for her roles in both indie films and major blockbusters. With standout performances in 'Lost in Translation' and as Black Widow in the Marvel Cinematic Universe, she has garnered critical acclaim. Johansson is also an advocate for women's rights and diverse representation in film."
}

# Define image file names
image_files = [
    "denzel.jpg",
    "scarlett.jpg",
    "robert.jpg",
    "sandra.jpg",
    "leo.jpg"
]

# Check if the image files exist
for image_file in image_files:
    if not os.path.isfile(image_file):
        st.error(f"Could not find image file: {image_file}")

# Characters for display
characters = list(class_dict.keys())

# Function to create circular images with fixed size and transparency
def create_circular_image(image_path, size=(200, 200)):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, size)  # Resize to fixed size
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, (255), thickness=-1)

    # Create a new image with an alpha channel (RGBA)
    circular_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    circular_img[:, :, 3] = 0  # Set alpha channel to 0 (fully transparent)
    circular_img[mask == 255, 3] = 255  # Set alpha channel to 255 (fully opaque) where the mask is white

    return circular_img  # Return the RGBA image

# Center the title using a smaller HTML heading
st.markdown("<h2 style='text-align: center;'>Celebrity Classification for Actor Recognition</h2>", unsafe_allow_html=True)

# Add space between the title and photos
st.markdown("<br><br>", unsafe_allow_html=True)  # Adjust the number of <br> for desired space

# Display circular images
if all(os.path.isfile(image_file) for image_file in image_files):
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            circular_image = create_circular_image(image_files[i])
            st.image(circular_image, caption=characters[i], use_column_width='auto')  # Maintain fixed size for images

# Function to crop the image if two eyes are detected
def get_cropped_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    return None

# Function to classify the cropped image
def classify_image(image):
    img_resized = cv2.resize(image, (32, 32))
    img_har = w2d(img_resized, 'db1', 5)

    combined_features = np.hstack((img_resized.flatten(), img_har.flatten()))
    combined_features = combined_features.reshape(1, -1)

    prediction = model_pipeline.predict(combined_features)
    return prediction[0] if prediction.size > 0 else None

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    image_cv = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    # Get the cropped face image
    cropped_face = get_cropped_face(image_cv)
    
    if cropped_face is not None:
        # If a face is detected and cropped, classify the image
        prediction_index = classify_image(cropped_face)

        if prediction_index is not None:
            predicted_class = list(class_dict.keys())[prediction_index]
            details = details_dict.get(predicted_class, "No details available.")

            # Create two columns for image and details
            col1, col2 = st.columns([2, 3])  # Adjust ratios for desired width

            with col1:
                st.image(cropped_face, caption=predicted_class, use_column_width=True)

            with col2:
                st.markdown(f"<h3>{predicted_class}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 18px;'>{details}</p>", unsafe_allow_html=True)  # Adjust font size as needed
        else:
            st.write("Could not detect a valid character.")
    else:
        st.warning("No face detected! Please upload an image containing a face for classification.")
