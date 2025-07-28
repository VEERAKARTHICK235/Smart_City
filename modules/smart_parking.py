import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_parking_model():
    """Trains a model to predict parking spot availability."""
    # 1. Mock Data: time_of_day, is_weekend (0/1), availability (0=Occupied, 1=Available)
    data = {
        'time_of_day': [8, 10, 12, 14, 16, 18, 10, 14, 18],
        'is_weekend': [0, 0, 0, 0, 0, 0, 1, 1, 1],
        'is_available': [1, 0, 0, 0, 1, 1, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    X = df[['time_of_day', 'is_weekend']]
    y = df['is_available']
    
    # 2. Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    print("🚗 Smart Parking Model Trained.")
    return model

def predict_parking_availability(model, hour, weekend):
    """Predicts if a parking spot is likely to be available."""
    prediction = model.predict([[hour, weekend]])
    proba = model.predict_proba([[hour, weekend]])
    
    status = "Available" if prediction[0] == 1 else "Occupied"
    confidence = proba[0][prediction[0]] * 100
    
    return f"Prediction: Spot is likely {status} with {confidence:.1f}% confidence."

# Example Usage:
# parking_model = train_parking_model()
# # Check weekday at 12 PM
# print(predict_parking_availability(parking_model, hour=12, weekend=0))
# # Check weekend at 6 PM
# print(predict_parking_availability(parking_model, hour=18, weekend=1))