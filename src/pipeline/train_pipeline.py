from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exceptions import CustomException
import sys

class TrainPipeline:
    def __init__(self):
        pass
    def start_training_pipeline(self):
        try:
            logging.info("Training pipeline started")
            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed")        
            transformation = DataTransformation()
            train_dataset, test_dataset, tokenizer = transformation.initiate_data_transformation(
                train_data_path,
                test_data_path
            )
            logging.info("Data transformation completed")   

            trainer = ModelTrainer()
            trainer.initiate_model_trainer(train_dataset,test_dataset,tokenizer)

            logging.info("Model training completed")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = TrainPipeline()
    obj.start_training_pipeline()