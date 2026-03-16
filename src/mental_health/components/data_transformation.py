import os
import sys
import pandas as pd
import numpy as np

from src.mental_health.logger import logger
from src.mental_health.exception import CustomException


class Data_Transformation:

    def __init__(self):
        self.train_data_path = "artifacts/train_dataset"
        self.test_data_path = "artifacts/test_dataset"

        self.transformed_train_data_path = "artifacts/transformed_train_dataset"
        self.transformed_test_data_path = "artifacts/transformed_test_dataset"


    def handle_outliers(self, df):
        """
        Handle outliers using IQR capping
        """

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


    def encode_categorical(self, df):

        categorical_cols = df.select_dtypes(include="object").columns

        for col in categorical_cols:
            df[col] = df[col].astype("category").cat.codes

        return df


    def transform_data(self, df):

        df = self.handle_outliers(df)

        df = self.encode_categorical(df)

        return df


    def data_transformation(self):

        try:

            logger.info("Starting Data Transformation")

            train_df = pd.read_csv(self.train_data_path)
            test_df = pd.read_csv(self.test_data_path)

            logger.info("Train and Test datasets loaded")

            transformed_train_df = self.transform_data(train_df)
            transformed_test_df = self.transform_data(test_df)

            os.makedirs("artifacts", exist_ok=True)

            transformed_train_df.to_csv(self.transformed_train_data_path, index=False)
            transformed_test_df.to_csv(self.transformed_test_data_path, index=False)

            logger.info("Transformed datasets saved successfully")

            return (
                self.transformed_train_data_path,
                self.transformed_test_data_path
            )

        except Exception as e:
            logger.error("Error during data transformation")
            raise CustomException(e, sys)