import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.mental_health.exception import CustomException
from src.mental_health.logger import logger
class DataIngestion:
    def __init__(self):
        self.raw_dataset_path = "artifacts/raw_dataset"
        self.train_dataset_path = "artifacts/train_dataset"
        self.test_dataset_path = "artifacts/test_dataset"

    
    def initiate_data_Igestion(self):
        logger.info("Initiate the Data Ingestion")
        try:
            dataset_path = "E:\Freelancing\ghaiga\mental health\data\Stress Indicators Dataset for Mental Health Classification.csv"
            # Read the dataset
            df = pd.read_csv(dataset_path)
            logger.info("Dataset read successfully!")
            # create the artifacts folder
            os.makedirs(os.path.dirname(self.raw_dataset_path),exist_ok=True)

            df.to_csv(self.raw_dataset_path,index=False)
            logger.info("raw file saved!")

            train_data,test_data = train_test_split(df, test_size=0.2,random_state=42)

            os.makedirs(os.path.dirname(self.train_dataset_path),exist_ok=True)

            train_data.to_csv(self.train_dataset_path,index=False)
            logger.info("Train data saved successfully!")

            os.makedirs(os.path.dirname(self.test_dataset_path),exist_ok=True)

            test_data.to_csv(self.test_dataset_path,index=False)
            logger.info("Test dataset saved successfully!")
            return (
                self.train_dataset_path,
                self.test_dataset_path
            )
        except Exception as e:
            raise CustomException(e,sys)
