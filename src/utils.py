import os
import sys
import pickle
import yaml

from src.logger import logging
from src.exceptions import CustomException


def read_yaml(file_path):
    """
    Read yaml file and return data
    """

    try:
        with open(file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)

            logging.info(f"YAML file loaded successfully from {file_path}")

            return data

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    """
    Save pickle object
    """

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load pickle object
    """

    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logging.info(f"Object loaded from {file_path}")

        return obj

    except Exception as e:
        raise CustomException(e, sys)


def create_directories(paths):
    """
    Create multiple directories
    """

    try:
        for path in paths:
            os.makedirs(path, exist_ok=True)

            logging.info(f"Created directory: {path}")

    except Exception as e:
        raise CustomException(e, sys)