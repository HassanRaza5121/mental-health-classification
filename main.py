from src.mental_health.logger import logger
from src.mental_health.exception import CustomException
#from src.mental_health.components.data_ingestion import DataIngestion
from src.mental_health.components.data_validation import Data_Validation
if __name__== "__main__":
    '''logger.info("Data Ingestion initiated!")
    ingestion = DataIngestion()
    ingestion.initiate_data_Igestion()
    logger.info('Data Ingestion completed')'''
    #config = DataValidationConfig()
    data_valid = Data_Validation()
    report = data_valid.data_validation()
    print(report)

    
