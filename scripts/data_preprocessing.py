import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    features = df.drop(columns=["CustomerId", "Surname", "Exited"])
    # One-hot encode categorical columns
    features = pd.get_dummies(features, columns=["Geography", "Gender"], drop_first=True)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(features)
    return pd.DataFrame(df_scaled, columns=features.columns)
