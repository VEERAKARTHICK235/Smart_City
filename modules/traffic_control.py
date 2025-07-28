import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor # <-- Import the refined model
from sklearn.metrics import mean_squared_error

def train_traffic_model():
    """
    Trains a Gradient Boosting model to predict traffic volume from a CSV file.
    """
    # 1. Load Real Data from CSV
    try:
        df = pd.read_csv('data/traffic_data.csv')
    except FileNotFoundError:
        print("Error: 'data/traffic_data.csv' not found. Please create it first.")
        return None

    # 2. Define Features (X) and Target (y)
    X = df[['time_of_day', 'day_of_week', 'weather']]
    y = df['traffic_volume']

    # 3. Split data and train a more advanced model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using a more robust model: GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate (optional)
    predictions = model.predict(X_test)
    print(f"🚦 Traffic Model (Gradient Boosting) MSE: {mean_squared_error(y_test, predictions):.2f}")
    
    return model

def predict_traffic(model, hour, day, weather):
    """Predicts traffic volume and suggests an action."""
    if model is None:
        return "Model not trained. Cannot make a prediction."
        
    prediction = model.predict([[hour, day, weather]])
    
    # Simple rule-based control
    if prediction[0] > 600:
        action = "High congestion expected. Extend green light duration."
    elif prediction[0] > 300:
        action = "Moderate traffic. Normal signal timing."
    else:
        action = "Light traffic. Consider shorter green light duration."
        
    return f"Predicted Volume: {int(prediction[0])} vehicles/hr. Action: {action}"