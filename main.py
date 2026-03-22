from src.mental_health.logger import logger
from src.mental_health.exception import CustomException
from src.mental_health.components.data_ingestion import DataIngestion
from src.mental_health.components.data_validation import Data_Validation
from src.mental_health.components.data_transformation import Data_Transformation
from src.mental_health.components.model_trainer import ModelTrainer
from src.mental_health.components.model_evaluation import ModelEvaluator

if __name__== "__main__":
    logger.info("Data Ingestion initiated!")
    ingestion = DataIngestion()
    ingestion.initiate_data_Igestion()
    logger.info('Data Ingestion completed')
    #config = DataValidationConfig()
    data_valid = Data_Validation()
    report = data_valid.data_validation()
    logger.info("Data Validation completed")
    logger.info("Data Transformation initiated!")
    data_transform = Data_Transformation()
    data_transform.data_transformation()
    logger.info("Data Transformation completed")

    logger.info("Model Training initiated!")
    model_trainer = ModelTrainer()
    model_trainer.train_all_models()
    logger.info("Model Training completed")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models()
