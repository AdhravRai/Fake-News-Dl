import os
import sys
import torch
import pandas as pd

from dataclasses import dataclass
from transformers import DistilBertTokenizerFast
from torch.utils.data import Dataset
from src.logger import logging
from src.exceptions import CustomException

@dataclass
class DataTransformationConfig:
    max_length:int=256

class NewsDataset(Dataset):
    def __init__(self,encodings,labels):
        self.encodings=encodings
        self.labels=labels

    def __getitem__(self,idx):
        item={}

        for key, value in self.encodings.items():
            item[key] = torch.tensor(value[idx])

        item["labels"] = torch.tensor(self.labels.iloc[idx])

        return item

    def __len__(self):
        return len(self.labels)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Data transformation started")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            X_train=train_df['text']
            X_test=test_df['text']
            y_train=train_df['label']
            y_test=test_df['label']
            train_encodings=self.tokenizer(X_train.tolist(),truncation=True,padding=True,max_length=self.data_transformation_config.max_length)
            test_encodings=self.tokenizer(X_test.tolist(),truncation=True,padding=True,max_length=self.data_transformation_config.max_length)
            
            train_dataset=NewsDataset(train_encodings,y_train)
            test_dataset=NewsDataset(test_encodings,y_test)
            logging.info("Data transformation completed")

            return(
                 train_dataset,
                 test_dataset,
                 self.tokenizer
            )
        except Exception as e:
            raise CustomException(e, sys)


        
