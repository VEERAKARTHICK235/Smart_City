import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_dataset(data, n_past):
    """Helper function to format data for LSTM."""
    X, y = [], []
    for i in range(n_past, len(data)):
        X.append(data[i - n_past:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def train_aqi_model():
    """Trains an LSTM model to predict AQI."""
    # 1. Mock Data: Hourly AQI readings
    aqi_data = np.array([50, 55, 60, 65, 70, 68, 62, 65, 75, 80, 90, 85, 80, 75]).reshape(-1, 1)
    
    # We will use the last 3 hours (n_past=3) to predict the next hour
    n_past = 3
    X, y = create_dataset(aqi_data, n_past)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 2. Build and Train LSTM Model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_past, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    print("🌬️ Air Quality LSTM Model Trained.")
    return model, n_past

def predict_aqi(model, n_past, last_n_hours_data):
    """Forecasts the next hour's AQI."""
    if len(last_n_hours_data) != n_past:
        return f"Error: Model requires exactly {n_past} data points."
    
    input_data = np.array(last_n_hours_data).reshape((1, n_past, 1))
    prediction = model.predict(input_data, verbose=0)
    return f"Predicted AQI for next hour: {prediction[0][0]:.1f}"

# Example Usage:
# aqi_model, n_steps = train_aqi_model()
# last_3_hours = [85, 80, 75]
# print(predict_aqi(aqi_model, n_steps, last_3_hours))