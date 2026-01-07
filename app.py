import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Human Face Detection App",
    layout="centered"
)

st.title("🧠 Human Face Detection using Streamlit")
st.markdown("Upload an image and adjust the parameters to detect human faces.")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("🔧 Detection Parameters")

scale_factor = st.sidebar.slider(
    "Scale Factor",
    min_value=1.05,
    max_value=1.50,
    value=1.10,
    step=0.05,
    help="Controls how much the image size is reduced at each image scale"
)

min_neighbors = st.sidebar.slider(
    "Min Neighbors",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="Higher value results in fewer detections but higher quality"
)

min_face_size = st.sidebar.slider(
    "Minimum Face Size (pixels)",
    min_value=30,
    max_value=200,
    value=50,
    step=10,
    help="Minimum face size to be detected"
)

# -----------------------------
# Load Haar Cascade Model
# -----------------------------
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

face_cascade = load_face_detector()

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read and display original image
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("🖼️ Uploaded Image Preview")
    st.image(image, use_container_width=True)

    # Convert to OpenCV format
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # -----------------------------
    # Face Detection
    # -----------------------------
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size)
    )

    # Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image_np,
            "Human face identified",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # -----------------------------
    # Display Result
    # -----------------------------
    st.subheader("✅ Face Detection Result")

    if len(faces) > 0:
        st.success(f"{len(faces)} human face(s) detected")
    else:
        st.warning("No human face detected")

    st.image(image_np, use_container_width=True)

else:
    st.info("Please upload an image to start face detection.")
