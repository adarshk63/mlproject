import sys
import os

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    """
    End-to-end training pipeline:
      - Performs data ingestion (train/test split)
      - Performs data transformation (feature engineering + preprocessing)
      - Trains the model
      - Saves trained model & preprocessor
    """

    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("======== Training Pipeline Started ========")

            # 1. Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed. Train: {train_path}, Test: {test_path}")

            # 2. Data Transformation
            transformer = DataTransformation()
            (
                train_arr,
                test_arr,
                preprocessor_path
            ) = transformer.initiate_data_transformation(
                train_path=train_path,
                test_path=test_path,
                target_column_name="Demand Forecast"   # IMPORTANT: replace with your true target column
            )

            logging.info(f"Data Transformation completed. Preprocessor saved to: {preprocessor_path}")

            # 3. Model Training
            model_trainer = ModelTrainer()
            model_report = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info("======== Training Pipeline Completed Successfully ========")
            logging.info(f"Model Report: {model_report}")

            return model_report

        except Exception as e:
            raise CustomException(e, sys)
