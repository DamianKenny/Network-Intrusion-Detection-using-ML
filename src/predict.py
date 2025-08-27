
import pandas as pd
import numpy as np
import joblib
import sys

def predict_attack_type(features):
    """
    Predict if a network connection is an attack or normal
    
    Parameters:
    features (list or array): Feature values for a single connection
    
    Returns:
    dict: Prediction results with probabilities
    """
    
    try:
        model = joblib.load('models/random_forest_binary.pkl')
        scaler = joblib.load('models/scaler_selected.pkl')
        selected_features = joblib.load('models/selected_features.pkl')
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please make sure you've trained the model first.")
        return None
    
    features_df = pd.DataFrame([features], columns=selected_features)
    
  
    features_scaled = scaler.transform(features_df)
    
   
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
  
    result = {
        'prediction': 'Attack' if prediction[0] == 1 else 'Normal',
        'confidence': float(probabilities[0][prediction[0]]),
        'probabilities': {
            'Normal': float(probabilities[0][0]),
            'Attack': float(probabilities[0][1])
        }
    }
    
    return result

if __name__ == "__main__":
  
    if len(sys.argv) > 1:
      
        features = [float(x) for x in sys.argv[1:]]
    else:
        
        features = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    

    selected_features = joblib.load('models/selected_features.pkl')
    if len(features) != len(selected_features):
        print(f"Error: Expected {len(selected_features)} features, got {len(features)}")
        sys.exit(1)
    
   
    result = predict_attack_type(features)
    
    if result:
        print("Prediction Results:")
        print(f"Type: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Probabilities:")
        print(f"  Normal: {result['probabilities']['Normal']:.4f}")
        print(f"  Attack: {result['probabilities']['Attack']:.4f}")
