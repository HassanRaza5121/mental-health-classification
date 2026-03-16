import os
import sys
import json
import pandas as pd
from src.mental_health.logger import logger
from src.mental_health.exception import CustomException


class Data_Validation:

    def __init__(self):
        self.train_data_path = "artifacts/train_dataset"
        self.test_data_path = "artifacts/test_dataset"
        self.report_path = "artifacts/data_validation_report.json"

    def validate_dataframe(self, df):

        report = {}

        # Missing values
        report["missing_values"] = df.isnull().sum().to_dict()

        # Data types
        report["data_types"] = df.dtypes.astype(str).to_dict()

        # Duplicate rows
        report["duplicate_rows"] = int(df.duplicated().sum())

        # Outlier detection using IQR
        numeric_df = df.select_dtypes(include="number")

        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1

        outliers = ((numeric_df < (Q1 - 1.5 * IQR)) |
                    (numeric_df > (Q3 + 1.5 * IQR))).sum()

        report["outliers"] = outliers.to_dict()

        return report

    def data_validation(self):

        try:

            logger.info("Starting Data Validation")

            # Load datasets
            train_df = pd.read_csv(self.train_data_path)
            test_df = pd.read_csv(self.test_data_path)

            logger.info("Train and Test datasets loaded")

            # Validate datasets
            train_report = self.validate_dataframe(train_df)
            test_report = self.validate_dataframe(test_df)

            final_report = {
                "train_data_validation": train_report,
                "test_data_validation": test_report
            }

            # Save report
            os.makedirs("artifacts", exist_ok=True)

            with open(self.report_path, "w") as f:
                json.dump(final_report, f, indent=4)

            logger.info("Data Validation Completed. Report saved.")

            return True

        except Exception as e:
            logger.error("Error occurred during data validation")
            raise CustomException(e, sys)