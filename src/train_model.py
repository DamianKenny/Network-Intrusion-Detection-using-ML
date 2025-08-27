import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_type='binary'):
    """
    Train and evaluate multiple models
    """
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Save the model
        joblib.dump(model, f'models/{name.replace(" ", "_").lower()}_{model_type}.pkl')
    
    return results

def plot_results(results, y_test, model_type='binary'):
    """
    Plot model comparison results
    """
    # Extract accuracies
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=accuracies, y=model_names)
    plt.title(f'Model Comparison ({model_type} classification)')
    plt.xlabel('Accuracy')
    plt.xlim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(accuracies):
        plt.text(v + 0.01, i, f'{v:.4f}', color='black', va='center')
    
    plt.tight_layout()
    plt.savefig(f'results/model_comparison_{model_type}.png')
    plt.show()
    
    # Plot confusion matrix for the best model
    best_model_name = model_names[np.argmax(accuracies)]
    best_result = results[best_model_name]
    
    cm = confusion_matrix(y_test, best_result['predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name} ({model_type})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{model_type}.png')
    plt.show()
    
    return model_names, accuracies

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load the preprocessed data (this should be done from files)
    try:
        X_train_scaled = joblib.load('data/processed/X_train_scaled.pkl')
        X_test_scaled = joblib.load('data/processed/X_test_scaled.pkl')
        y_train_binary = joblib.load('data/processed/y_train_binary.pkl')
        y_test_binary = joblib.load('data/processed/y_test_binary.pkl')
        y_train_multiclass = joblib.load('data/processed/y_train_multiclass.pkl')
        y_test_multiclass = joblib.load('data/processed/y_test_multiclass.pkl')
        
        print("Loaded preprocessed data from files.")
        
    except FileNotFoundError:
        print("Processed data files not found. Please run the preprocessing steps first.")
        exit(1)
    
    # Train and evaluate binary classification models
    print("Binary Classification Models:")
    binary_results = train_and_evaluate_models(
        X_train_scaled, 
        X_test_scaled, 
        y_train_binary, 
        y_test_binary,
        'binary'
    )
    
    # Plot binary results
    binary_models, binary_accuracies = plot_results(binary_results, y_test_binary, 'binary')
    
    # Train and evaluate multiclass classification models
    print("\nMulticlass Classification Models:")
    multiclass_results = train_and_evaluate_models(
        X_train_scaled, 
        X_test_scaled, 
        y_train_multiclass, 
        y_test_multiclass,
        'multiclass'
    )
    
    # Plot multiclass results
    multiclass_models, multiclass_accuracies = plot_results(multiclass_results, y_test_multiclass, 'multiclass')
    
    # Save results summary
    results_summary = pd.DataFrame({
        'Model': binary_models + multiclass_models,
        'Accuracy': binary_accuracies + multiclass_accuracies,
        'Type': ['Binary'] * len(binary_models) + ['Multiclass'] * len(multiclass_models)
    })
    
    results_summary.to_csv('results/model_results_summary.csv', index=False)
    print("Training completed! Results saved to results/ directory.")

from google.colab import files
files.download('src/train_model.py')

# Create and download the prediction script
%%writefile src/predict.py
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
    # Load the trained model and preprocessing objects
    try:
        model = joblib.load('models/random_forest_binary.pkl')
        scaler = joblib.load('models/scaler_selected.pkl')
        selected_features = joblib.load('models/selected_features.pkl')
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please make sure you've trained the model first.")
        return None
    
    # Convert features to DataFrame with correct column names
    features_df = pd.DataFrame([features], columns=selected_features)
    
    # Scale the features
    features_scaled = scaler.transform(features_df)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    # Return results
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
    # Example usage
    if len(sys.argv) > 1:
        # Read features from command line
        features = [float(x) for x in sys.argv[1:]]
    else:
        # Use sample features (replace with actual values)
        features = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Ensure we have the right number of features
    selected_features = joblib.load('models/selected_features.pkl')
    if len(features) != len(selected_features):
        print(f"Error: Expected {len(selected_features)} features, got {len(features)}")
        sys.exit(1)
    
    # Make prediction
    result = predict_attack_type(features)
    
    if result:
        print("Prediction Results:")
        print(f"Type: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Probabilities:")
        print(f"  Normal: {result['probabilities']['Normal']:.4f}")
        print(f"  Attack: {result['probabilities']['Attack']:.4f}")

files.download('src/predict.py')

# Create and download requirements.txt
!pip freeze > requirements.txt
files.download('requirements.txt')

# Create and download .gitignore
%%writefile .gitignore
__pycache__/
*.pyc
*.pkl
*.model
data/processed/
results/
.DS_Store
.ipynb_checkpoints/
venv/
.env
*.log

files.download('.gitignore')

print("All files have been created and downloaded successfully!")
