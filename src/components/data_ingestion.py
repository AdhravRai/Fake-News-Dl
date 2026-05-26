import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exceptions import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    source_data_path: str = os.path.join(
        "notebooks/data",
        "processed_news.csv"
    )

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")

        try:
            df = pd.read_csv(self.ingestion_config.source_data_path)
            df = df.sample(2000, random_state=42)
            os.makedirs("artifacts",exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Train test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Data Ingestion Completed")
            return(
                 self.ingestion_config.train_data_path,
                 self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

        
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    print(train_data)
    print(test_data)

            
    
