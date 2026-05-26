import os
import sys
import numpy as np

from dataclasses import dataclass
from transformers import (
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score, f1_score

from src.logger import logging
from src.exceptions import CustomException


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions,axis=1)
        accuracy = accuracy_score(labels, preds)

        f1 = f1_score(labels, preds)

        return {
            "accuracy": accuracy,
            "f1": f1
        }

    def initiate_model_trainer(self,train_dataset,test_dataset,tokenizer):
        try:
            logging.info("Model training started")

            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=2
            )
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=1,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_dir="./logs",
                logging_steps=10,
                load_best_model_at_end=True
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,                
                compute_metrics=self.compute_metrics
            )
            trainer.train()
            results = trainer.evaluate()

            model.save_pretrained(self.model_trainer_config.trained_model_path)
            tokenizer.save_pretrained(self.model_trainer_config.trained_model_path)

            logging.info("Model training completed")

            return results

        except Exception as e:
            raise CustomException(e, sys)