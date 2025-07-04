import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from data_preprocessing import load_and_preprocess_data

def train_and_save_churn_model(data_path, model_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    X = load_and_preprocess_data(data_path)
    y = df['Exited']
    
    # Split data (optional, for demonstration)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_and_save_churn_model("data/customer_data.csv", "models/churn_model.pkl")
