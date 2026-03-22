import pickle
import shap
import pandas as pd

# Load model & test data
model = pickle.load(open("artifacts/models/xgboost.pkl", "rb"))
X_test = pd.read_csv("artifacts/transformed_test_dataset.csv").drop(columns=["stress_experience"])
y_test = pd.read_csv("artifacts/transformed_test_dataset.csv")["stress_experience"]

# Load scaler
scaler = pickle.load(open("artifacts/models/scaler.pkl", "rb"))
X_scaled = scaler.transform(X_test)

# Wrapper function
def model_predict(X):
    return model.predict_proba(X)

# Use SHAP with callable
explainer = shap.Explainer(model_predict, X_scaled)
shap_values = explainer(X_scaled)

shap.summary_plot(shap_values, X_scaled)