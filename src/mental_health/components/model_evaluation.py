import os
import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from src.mental_health.logger import logger
from src.mental_health.exception import CustomException

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
# ==============================
# HELPER FUNCTION
# ==============================
def plot_confusion_matrix(y_test, y_pred, model_name):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
class ModelEvaluator:

    def __init__(self):
        self.test_data_path = "artifacts/transformed_test_dataset.csv"
        self.model_dir = "artifacts/models"

    def evaluate_model(self, model, X_test, y_test):

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return acc, report, y_pred

    def evaluate_all_models(self):

        try:
            logger.info("Starting Model Evaluation")

            test_df = pd.read_csv(self.test_data_path)
            target_column = "stress_experience"

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # ✅ Load scaler
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            # ✅ Transform + keep column names
            X_scaled = scaler.transform(X_test)
            X_scaled = pd.DataFrame(X_scaled, columns=X_test.columns)

            results = {}

            for file in os.listdir(self.model_dir):

                if file.endswith(".pkl"):

                    # ❌ Skip scaler
                    if file == "scaler.pkl":
                        continue

                    model_path = os.path.join(self.model_dir, file)

                    with open(model_path, "rb") as f:
                        model = pickle.load(f)

                    # ✅ Extra safety (skip non-model objects)
                    if not hasattr(model, "predict"):
                        continue

                    model_name = file.replace(".pkl", "")

                    acc, report, y_pred = self.evaluate_model(model, X_scaled, y_test)
                    plot_confusion_matrix(y_test, y_pred, model_name)

                    results[model_name] = {
                        "accuracy": acc,
                        "f1_score": report["weighted avg"]["f1-score"],
                        "precision": report["weighted avg"]["precision"],
                        "recall": report["weighted avg"]["recall"]
                    }

                    logger.info(f"{model_name} evaluated")

            # ✅ Sort results
            results_df = pd.DataFrame(results).T.sort_values(by="f1_score", ascending=False)

            print("\n===== MODEL PERFORMANCE =====\n")
            print(results_df)
            results_df.plot(kind="bar", figsize=(10,5))
            plt.title("Model Comparison")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # ✅ Save results
            results_df.to_csv("artifacts/model_performance.csv")

            return results_df

        except Exception as e:
            raise CustomException(e, sys)