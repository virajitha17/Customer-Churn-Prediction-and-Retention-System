import pandas as pd

def segment_customers(data, churn_probs, threshold=0.5):
    segments = pd.DataFrame(data.copy())
    segments["ChurnProbability"] = churn_probs
    segments["Segment"] = segments["ChurnProbability"].apply(
        lambda x: "High Risk" if x >= threshold else "Low Risk")
    return segments