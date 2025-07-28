import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# You can reuse the create_dataset helper function from air_quality.py
from .air_quality import create_dataset 

def train_energy_model():
    """Trains an LSTM model to predict energy demand."""
    # 1. Mock Data: Hourly energy demand (in Megawatts)
    energy_data = np.array([200, 210, 220, 250, 300, 320, 310, 280, 260, 240, 230]).reshape(-1, 1)
    
    n_past = 3
    X, y = create_dataset(energy_data, n_past)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 2. Build and Train LSTM Model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_past, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    print("⚡ Energy Grid LSTM Model Trained.")
    return model, n_past

def predict_energy_demand(model, n_past, last_n_hours_demand):
    """Forecasts the next hour's energy demand."""
    input_data = np.array(last_n_hours_demand).reshape((1, n_past, 1))
    prediction = model.predict(input_data, verbose=0)
    return f"Predicted Energy Demand for next hour: {prediction[0][0]:.1f} MW"

# Example Usage:
# energy_model, n_steps = train_energy_model()
# last_3_hours_demand = [260, 240, 230]
# print(predict_energy_demand(energy_model, n_steps, last_3_hours_demand))