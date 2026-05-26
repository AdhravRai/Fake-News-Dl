import sys
import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from src.exceptions import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        self.model_path = "artifacts/model"
        self.tokenizer_path = "artifacts/tokenizer"
        logging.info("Loading tokenizer and model")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.tokenizer_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)

    def predict(self, text):
        try:
            logging.info("Tokenizing input text")
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )
            logging.info("Making prediction")

            with torch.no_grad():
                outputs = self.model(**inputs)

            prediction = torch.argmax(outputs.logits, dim=1).item()
            if prediction == 1:
                return "REAL NEWS"
            else:
                return "FAKE NEWS"

        except Exception as e:
            raise CustomException(e, sys)