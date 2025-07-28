import pandas as pd
from sklearn.linear_model import LinearRegression

def train_waste_model():
    """Trains a model to predict bin fill level."""
    # 1. Mock Data: days_since_collection, location_type (0=residential, 1=commercial), fill_level (%)
    data = {
        'days_since_collection': [1, 2, 3, 1, 2, 3, 1, 2, 3, 4],
        'location_type': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        'fill_level': [20, 45, 65, 30, 65, 95, 22, 48, 70, 98]
    }
    df = pd.DataFrame(data)

    # 2. Define Features (X) and Target (y)
    X = df[['days_since_collection', 'location_type']]
    y = df['fill_level']
    
    # 3. Train model
    model = LinearRegression()
    model.fit(X, y)
    print("🗑️ Waste Management Model Trained.")
    return model

def predict_fill_level(model, days, loc_type):
    """Predicts bin fill level and suggests action."""
    level = model.predict([[days, loc_type]])[0]
    if level >= 85:
        return f"Predicted Fill Level: {level:.1f}%. Action: Schedule collection now."
    else:
        return f"Predicted Fill Level: {level:.1f}%. Action: No collection needed yet."

# Example Usage:
# waste_model = train_waste_model()
# print(predict_fill_level(waste_model, days=3, loc_type=1)) # Commercial area after 3 days