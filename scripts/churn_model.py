import pickle
import numpy as np

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_churn(model, data):
    return model.predict_proba(data)[:, 1]

import pandas as pd

def segment_customers(data, churn_probs, threshold=0.5):
    segments = pd.DataFrame(data.copy())
    segments["ChurnProbability"] = churn_probs
    segments["Segment"] = segments["ChurnProbability"].apply(
        lambda x: "High Risk" if x >= threshold else "Low Risk")
    return segments
