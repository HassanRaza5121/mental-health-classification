import os
import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from src.mental_health.logger import logger
from src.mental_health.exception import CustomException


class Data_Transformation:

    def __init__(self):
        self.train_data_path = "artifacts/train_dataset"
        self.test_data_path = "artifacts/test_dataset"

        self.transformed_train_data_path = "artifacts/transformed_train_dataset.csv"
        self.transformed_test_data_path = "artifacts/transformed_test_dataset.csv"

        self.encoder_path = "artifacts/encoder.pkl"

    def handle_outliers(self, df):
        numeric_cols = df.select_dtypes(include="number").columns

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

        return df

    def data_transformation(self):

        try:
            logger.info("Starting Data Transformation")

            train_df = pd.read_csv(self.train_data_path)
            test_df = pd.read_csv(self.test_data_path)

            target_column = "stress_experience"

            # ✅ Separate features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # ✅ Handle outliers only on features
            X_train = self.handle_outliers(X_train)
            X_test = self.handle_outliers(X_test)

            # ✅ Encode categorical columns properly
            categorical_cols = X_train.select_dtypes(include="object").columns

            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

            X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
            X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

            # ✅ Fix labels for models (IMPORTANT)
            y_train = y_train.astype(int) - 1
            y_test = y_test.astype(int) - 1

            # ✅ Save encoder
            os.makedirs("artifacts", exist_ok=True)
            import pickle
            with open(self.encoder_path, "wb") as f:
                pickle.dump(encoder, f)

            # ✅ Combine back
            train_processed = pd.concat([X_train, y_train], axis=1)
            test_processed = pd.concat([X_test, y_test], axis=1)

            train_processed.to_csv(self.transformed_train_data_path, index=False)
            test_processed.to_csv(self.transformed_test_data_path, index=False)

            logger.info("Data Transformation Completed")

            return (
                self.transformed_train_data_path,
                self.transformed_test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)