import streamlit as st
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import os

# --- Import Prediction Functions ---
from modules import traffic_control, waste_management, air_quality, water_management, energy_grid, public_safety, smart_parking, citizen_feedback

# --- Helper Function to Load All Models ---
@st.cache_resource
def load_all_models():
    """Load all trained models from the /models directory."""
    models = {}
    models['traffic'] = joblib.load('models/traffic_model.pkl')
    models['waste'] = joblib.load('models/waste_model.pkl')
    models['aqi'] = load_model('models/aqi_model.keras')
    models['water'] = joblib.load('models/anomaly_model.pkl')
    models['energy'] = load_model('models/energy_model.keras')
    models['parking'] = joblib.load('models/parking_model.pkl')
    return models

# --- Main Dashboard ---
st.set_page_config(layout="wide")
st.title("🏙️ Smart City Interactive Dashboard")

# Check if models exist
if not all(os.path.exists(f'models/{f}') for f in ['traffic_model.pkl', 'aqi_model.keras']):
    st.error("Models not found! Please run `python main.py` first to train and save the models.")
else:
    # Load models
    models = load_all_models()
    
    # --- Sidebar for User Input ---
    st.sidebar.title("Simulation Controls")

    # --- Layout ---
    col1, col2 = st.columns(2)

    with col1:
        st.header("🚦 Smart Traffic Control")
        hour = st.sidebar.slider("Time of Day (Hour)", 0, 23, 17)
        day = st.sidebar.selectbox("Day of Week", options=[1,2,3,4,5,6,7], format_func=lambda x: {1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',7:'Sun'}[x])
        weather = st.sidebar.selectbox("Weather", options=[0, 1], format_func=lambda x: "Clear" if x == 0 else "Rain")
        traffic_prediction = traffic_control.predict_traffic(models['traffic'], hour, day, weather)
        st.info(traffic_prediction)

        st.header("💧 Smart Water Management")
        flow = st.sidebar.number_input("Current Water Flow (L/min)", min_value=5.0, max_value=100.0, value=10.2, step=0.1)
        anomaly_prediction = water_management.detect_anomaly(models['water'], flow)
        if "Anomaly" in anomaly_prediction:
            st.error(anomaly_prediction)
        else:
            st.success(anomaly_prediction)
        
        st.header("🚓 Public Safety (CCTV)")
        uploaded_file = st.file_uploader("Upload a CCTV Image", type=["jpg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded CCTV Frame")
            detections = public_safety.detect_objects_mock(uploaded_file.name)
            st.write(f"**Detected {len(detections)} objects:**")
            st.json([d['label'] for d in detections])


    with col2:
        st.header("🗑️ Smart Waste Management")
        days = st.sidebar.slider("Days Since Last Collection", 1, 7, 3)
        loc_type = st.sidebar.selectbox("Location Type", options=[0, 1], format_func=lambda x: "Residential" if x == 0 else "Commercial")
        waste_prediction = waste_management.predict_fill_level(models['waste'], days, loc_type)
        st.info(waste_prediction)

        st.header("🚗 Smart Parking")
        parking_hour = st.sidebar.slider("Parking Time (Hour)", 0, 23, 14)
        is_weekend = st.sidebar.selectbox("Is it a weekend?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        parking_prediction = smart_parking.predict_parking_availability(models['parking'], parking_hour, is_weekend)
        st.info(parking_prediction)

        st.header("📣 Citizen Feedback")
        feedback_text = st.text_area("Enter citizen feedback here:", "The new park is beautiful and the streets are very clean now, great job!")
        if st.button("Analyze Feedback"):
            analysis = citizen_feedback.analyze_feedback(feedback_text)
            st.write(f"**Sentiment:** {analysis['sentiment']}")
            st.write(f"**Suggested Action:** {analysis['summary']}")