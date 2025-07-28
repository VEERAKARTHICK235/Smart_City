import pandas as pd
from sklearn.ensemble import IsolationForest

def train_anomaly_detector():
    """Trains an Isolation Forest model to find anomalies."""
    # 1. Mock Data: Hourly water flow (liters/min). A few high values are anomalies.
    data = {'water_flow': [10.2, 10.5, 10.1, 10.3, 55.0, 10.4, 9.9, 9.8, 62.5, 10.0]}
    df = pd.DataFrame(data)

    # 2. Train Model
    model = IsolationForest(contamination=0.2, random_state=42) # Expect ~20% anomalies
    model.fit(df[['water_flow']])
    print("💧 Water Anomaly Detector Trained.")
    return model

def detect_anomaly(model, current_flow):
    """Detects if a given water flow is an anomaly."""
    prediction = model.predict([[current_flow]])
    if prediction[0] == -1:
        return f"Flow: {current_flow} L/min. Status: Anomaly Detected! Possible Leak."
    else:
        return f"Flow: {current_flow} L/min. Status: Normal."
        
# Example Usage:
# anomaly_model = train_anomaly_detector()
# print(detect_anomaly(anomaly_model, 10.2)) # Normal
# print(detect_anomaly(anomaly_model, 60.0)) # Anomaly