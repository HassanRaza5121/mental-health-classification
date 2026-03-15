from src.mental_health.logger import logger
from src.mental_health.exception import CustomException
from src.mental_health.components.data_ingestion import DataIngestion
if __name__== "__main__":
    logger.info("Data Ingestion initiated!")
    ingestion = DataIngestion()
    ingestion.initiate_data_Igestion()
    logger.info('Data Ingestion completed')
    
