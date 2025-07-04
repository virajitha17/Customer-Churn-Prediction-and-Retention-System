
from scripts.data_preprocessing import load_and_preprocess_data
from scripts.churn_model import load_model, predict_churn
from scripts.segmentation import segment_customers
from scripts.strategy_assign import assign_retention_strategy

if __name__ == "__main__":
    data_path = "data/customer_data.csv"
    data = load_and_preprocess_data(data_path)
    model = load_model("models/churn_model.pkl")
    predictions = predict_churn(model, data)
    segments = segment_customers(data, predictions)
    strategies = assign_retention_strategy(segments)

    print("Retention Strategies Assigned:\n", strategies.head())