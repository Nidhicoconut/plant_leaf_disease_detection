import streamlit as st
import os
import uuid
import base64
from PIL import Image
import joblib
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from db import create_tables, insert_user, validate_user, save_prediction

# ----------------- Background -----------------
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("static/photo-1637070773929-054cf3288cbb.jpeg")

# ----------------- Load Models -----------------
try:
    ml_model = joblib.load("models/ml_model.pkl")
    label_dict = joblib.load("models/label_dict.pkl")
    cnn_model = load_model("models/cnn_model.h5")
    qml_model = joblib.load("models/qml_model.pkl")
    qml_encoder = joblib.load("models/qml_encoder.pkl")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Quantum NN model with Qiskit 1.0 compatibility
try:
    # Import Qiskit components with version check
    import qiskit
    from qiskit.circuit.library import TwoLocal
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
    
    # Verify Qiskit version
    if not qiskit.__version__.startswith('1.'):
        raise ImportError("Qiskit version must be 1.x")
    
    class QuantumClassifier(nn.Module):
        def __init__(self, num_classes=len(label_dict)):
            super(QuantumClassifier, self).__init__()
            num_qubits = 8  # Matches 2x2x2 histogram features
            ansatz = TwoLocal(num_qubits, 'ry', 'cz', reps=1)
            qnn = EstimatorQNN(
                circuit=ansatz,
                input_params=ansatz.parameters[:num_qubits],
                weight_params=ansatz.parameters[num_qubits:]
            )
            self.qnn = TorchConnector(qnn)
            self.classifier = nn.Linear(1, num_classes)

        def forward(self, x):
            q_out = self.qnn(x)
            return self.classifier(q_out)

    # Initialize and load QNN model
    qnn_classifier = QuantumClassifier()
    
    # Load state dict with strict=False to handle potential architecture differences
    state_dict = torch.load("models/qnn_model.pth", map_location=torch.device('cpu'))
    qnn_classifier.load_state_dict(state_dict, strict=False)
    qnn_classifier.eval()
    
    st.success("✅ QNN model loaded successfully")
    
except Exception as e:
    qnn_classifier = None
    st.warning(f"⚠️ QNN model not loaded: {str(e)}")
    st.info("Note: QNN predictions will not be available. Other models will work normally.")

# ----------------- Feature Extraction -----------------
def extract_features_ml(image):
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (128, 128))
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.reshape(1, -1)

def extract_features_qml(image):
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (64, 64))
    hist = cv2.calcHist([image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.reshape(1, -1)

def extract_features_qnn(image):
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (64, 64))
    hist = cv2.calcHist([image], [0, 1, 2], None, [2, 2, 2], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

def prepare_dl_image(image):
    image = image.resize((128, 128)).convert("RGB")
    img_array = keras_image.img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def is_healthy_label(label):
    label = str(label).lower()
    return "healthy" in label or "no disease" in label

# ----------------- Prediction -----------------
def predict_with_ml(image):
    features = extract_features_ml(image)
    prediction = ml_model.predict(features)[0]
    label = label_dict[prediction] if isinstance(label_dict, dict) else label_dict.inverse_transform([prediction])[0]
    return "Healthy" if is_healthy_label(label) else "Diseased"

def predict_with_cnn(image):
    data = prepare_dl_image(image)
    prediction = cnn_model.predict(data)
    predicted_class = np.argmax(prediction)
    label = label_dict[predicted_class] if isinstance(label_dict, dict) else label_dict.inverse_transform([predicted_class])[0]
    return "Healthy" if is_healthy_label(label) else "Diseased"

def predict_with_qml(image):
    try:
        features = extract_features_qml(image)
        prediction = qml_model.predict(features)[0]
        label = qml_encoder.inverse_transform([prediction])[0]
        return "Healthy" if is_healthy_label(label) else "Diseased"
    except Exception as e:
        st.error(f"QML error: {e}")
        return "Error"

def predict_with_qnn(image):
    if qnn_classifier is None:
        return "Model not loaded"
    try:
        features = extract_features_qnn(image)
        features = StandardScaler().fit_transform(features.reshape(1, -1))
        features_tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            output = qnn_classifier(features_tensor)
            _, predicted = torch.max(output.data, 1)
        index = predicted.item()
        label = label_dict[index] if isinstance(label_dict, dict) else label_dict.inverse_transform([index])[0]
        return "Healthy" if is_healthy_label(label) else "Diseased"
    except Exception as e:
        st.error(f"QNN error: {e}")
        return "Error"

# ----------------- UI Logic -----------------
create_tables()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.show_login = False
    st.session_state.show_signup = False

st.markdown('<h1 style="color:white; text-shadow:2px 2px 4px #000;">🌿 Plant Leaf Disease Detection</h1>', unsafe_allow_html=True)

if not st.session_state.logged_in:
    tab1, tab2 = st.columns(2)
    
    # Login option
    with tab1:
        st.subheader("🔐 Login")
        if st.button("Login"):
            st.session_state.show_login = True
            st.session_state.show_signup = False

    # Sign Up option
    with tab2:
        st.subheader("🆕 Sign Up")
        if st.button("Sign Up"):
            st.session_state.show_signup = True
            st.session_state.show_login = False
    
    # Show the login form if the user clicks "Login"
    if st.session_state.show_login:
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Submit Login"):
            if validate_user(user, pwd):
                st.session_state.logged_in = True
                st.session_state.username = user
                st.session_state.show_login = False  # Hide login form after successful login
                st.rerun()
            else:
                st.error("Invalid credentials.")
    
    # Show the signup form if the user clicks "Sign Up"
    if st.session_state.show_signup:
        new_user = st.text_input("New Username")
        new_pwd = st.text_input("New Password", type="password")
        if st.button("Submit Signup"):
            try:
                insert_user(new_user, new_pwd)
                st.success("Registration successful.")
                st.session_state.show_signup = False  # Hide signup form after successful signup
                st.rerun()
            except Exception as e:
                st.error(f"Signup failed: {e}")
else:
    st.subheader(f"Welcome, {st.session_state.username}!")
    st.markdown("This app detects plant leaf diseases using ML, DL, QML, and QNN models.")
    # Remaining part of the code here...


    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

        if st.button("🧠 Predict"):
            with st.spinner("Analyzing..."):
                ml_result = predict_with_ml(image)
                cnn_result = predict_with_cnn(image)
                qml_result = predict_with_qml(image)
                qnn_result = predict_with_qnn(image) if qnn_classifier is not None else "Not available"

                results = {
                    "ML": ml_result,
                    "CNN": cnn_result,
                    "QML": qml_result,
                    "QNN": qnn_result if qnn_classifier is not None else "Not available"
                }

                # Only consider available predictions for final result
                valid_results = [v for v in results.values() if v in ["Healthy", "Diseased"]]
                final_result = max(set(valid_results), key=valid_results.count) if valid_results else "Uncertain"

                st.markdown("### 🔍 Model Predictions")
                for model, result in results.items():
                    icon = "✅" if result == "Healthy" else "❌" if result == "Diseased" else "⚠️"
                    st.write(f"{icon} **{model}**: {result}")

                if final_result == "Healthy":
                    st.success("🌿 Final Diagnosis: Healthy")
                elif final_result == "Diseased":
                    st.error("🚨 Final Diagnosis: Diseased")
                else:
                    st.warning("🤔 Final Diagnosis: Inconclusive")

                os.makedirs("uploads", exist_ok=True)
                image_path = os.path.join("uploads", f"{uuid.uuid4()}.jpg")
                image.save(image_path)
                save_prediction(
                    st.session_state.username, 
                    image_path, 
                    ml_result, 
                    cnn_result, 
                    qml_result, 
                    qnn_result if qnn_classifier is not None else "Not available", 
                    final_result
                )

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()