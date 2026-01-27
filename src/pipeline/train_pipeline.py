import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("=== Training Pipeline Started ===")
            ingestion_config = DataIngestionConfig()
            ingestion = DataIngestion(ingestion_config=ingestion_config)

            train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed. Train: {train_path}, Test: {test_path}")

       
            transformation_config = DataTransformationConfig()
            transformer = DataTransformation(config=transformation_config)


            train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
                train_path=train_path,
                test_path=test_path,
                target_column_name="Units Sold"
            )
            logging.info(f"Data Transformation completed. Preprocessor saved at: {preprocessor_path}")
            model_trainer = ModelTrainer()


            model_report = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info("======== Training Pipeline Completed Successfully ========")
            logging.info(f"Model Report: {model_report}")

            return model_report

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
