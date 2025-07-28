import joblib # Used to save and load models
import os

# Import all your modules
from modules import traffic_control, waste_management, air_quality, water_management, energy_grid, public_safety, smart_parking, citizen_feedback

# Create a directory to save models if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

def main():
    print("--- 🏙️  Welcome to the Smart City Project Dashboard 🏙️  ---")
    
    # --- 1. Train All Models and Save Them ---
    print("\n--- Training Models ---")
    
    # Traffic
    traffic_model = traffic_control.train_traffic_model()
    joblib.dump(traffic_model, 'models/traffic_model.pkl')

    # Waste
    waste_model = waste_management.train_waste_model()
    joblib.dump(waste_model, 'models/waste_model.pkl')
    
    # Air Quality
    aqi_model, aqi_n_steps = air_quality.train_aqi_model()
    aqi_model.save('models/aqi_model.keras') # Keras models have a specific save method

    # Water
    anomaly_model = water_management.train_anomaly_detector()
    joblib.dump(anomaly_model, 'models/anomaly_model.pkl')

    # Energy
    energy_model, energy_n_steps = energy_grid.train_energy_model()
    energy_model.save('models/energy_model.keras')

    # Parking
    parking_model = smart_parking.train_parking_model()
    joblib.dump(parking_model, 'models/parking_model.pkl')

    print("\n✅ All models trained and saved successfully!")

    # --- 2. Run Example Predictions ---
    print("\n--- Running Live Simulations ---")

    print("\n🚦 Smart Traffic Control:")
    print(traffic_control.predict_traffic(traffic_model, hour=18, day=5, weather=1))

    print("\n🗑️ Smart Waste Management:")
    print(waste_management.predict_fill_level(waste_model, days=3, loc_type=1))

    print("\n🌬️ Air Quality Monitoring:")
    print(air_quality.predict_aqi(aqi_model, aqi_n_steps, last_n_hours_data=[85, 80, 75]))

    print("\n💧 Smart Water Management:")
    print(water_management.detect_anomaly(anomaly_model, current_flow=60.0))

    print("\n⚡ Smart Energy Grid:")
    print(energy_grid.predict_energy_demand(energy_model, energy_n_steps, last_n_hours_demand=[260, 240, 230]))

    print("\n🚓 Public Safety (CCTV):")
    detections = public_safety.detect_objects_mock('data/cctv_feed_frame.jpg')
    print(f"Detected {len(detections)} objects, including '{detections[0]['label']}'.")

    print("\n🚗 Smart Parking:")
    print(smart_parking.predict_parking_availability(parking_model, hour=12, weekend=0))

    print("\n📣 Citizen Feedback Portal:")
    feedback = "The traffic light on main street is broken again, causing huge delays."
    analysis = citizen_feedback.analyze_feedback(feedback)
    print(f"Sentiment: {analysis['sentiment']} -> Action: {analysis['summary']}")


if __name__ == "__main__":
    main()