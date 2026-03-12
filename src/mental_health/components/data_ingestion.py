import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.mental_health.exception import CustomException
from src.mental_health.logger import logging


class DataIngestion:
    def __init__(self):
        self.raw_data_path = "artifacts/raw_data.csv"
        self.train_data_path = "artifacts/train.csv"
        self.test_data_path = "artifacts/test.csv"

    def initiate_data_ingestion(self):
        logging.info("Starting Data Ingestion process")

        try:
            # Example dataset path
            dataset_path = "data/mental_health.csv"

            # Read dataset
            df = pd.read_csv(dataset_path)
            logging.info("Dataset read successfully")

            # Create artifacts folder
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.raw_data_path, index=False)
            logging.info("Raw data saved in artifacts folder")

            # Train-test split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save train and test data
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logging.info("Train and Test data saved")

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)