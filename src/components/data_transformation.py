import sys
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
   
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    def _ensure_date_and_create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                if df["Date"].isna().any():
                    dates = pd.date_range(start="2020-01-01", periods=len(df), freq="D")
                    df["Date"] = df["Date"].fillna(pd.Series(dates, index=df.index))
            else:
                df["Date"] = pd.date_range(start="2020-01-01", periods=len(df), freq="D")

            # create time features
            df["year"] = df["Date"].dt.year
            df["month"] = df["Date"].dt.month
            df["day"] = df["Date"].dt.day
            df["weekday"] = df["Date"].dt.weekday  # Monday=0
            df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
            df["month_start"] = df["Date"].dt.is_month_start.astype(int)
            df["month_end"] = df["Date"].dt.is_month_end.astype(int)

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def _create_lag_feature(self, df: pd.DataFrame, target_col: str, lag: int = 1) -> pd.DataFrame:
        
        try:
            lag_col = f"prev_{target_col}"
            df[lag_col] = df[target_col].shift(lag)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def _detect_column_types(
        self, df: pd.DataFrame, exclude: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Infer numerical and categorical columns from dataframe, excluding specified columns.
        Treats object/category dtype as categorical; numeric dtypes as numerical.
        """
        try:
            exclude = exclude or []
            cols = [c for c in df.columns if c not in exclude]

            categorical_cols = [
                c
                for c in cols
                if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])
            ]

            numerical_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

            logging.info(f"Auto-detected numerical columns: {numerical_cols}")
            logging.info(f"Auto-detected categorical columns: {categorical_cols}")

            return numerical_cols, categorical_cols
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(
        self, numerical_columns: List[str], categorical_columns: List[str]
    ) -> ColumnTransformer:
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            transformers = []
            if numerical_columns:
                transformers.append(("num_pipeline", num_pipeline, numerical_columns))
            if categorical_columns:
                transformers.append(("cat_pipeline", cat_pipeline, categorical_columns))

            preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str,
        target_column_name: str = "Demand Forecast",
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
    ):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Ensure date features and lag exist on both train and test (apply same transform)
            # Important: create lag AFTER target exists in each df; notebook used shift(1) then dropna
            train_df = self._ensure_date_and_create_time_features(train_df)
            test_df = self._ensure_date_and_create_time_features(test_df)

            if target_column_name not in train_df.columns:
                raise ValueError(f"Target column '{target_column_name}' not found in train file.")
            if target_column_name not in test_df.columns:
                raise ValueError(f"Target column '{target_column_name}' not found in test file.")

            # Create lag feature on each df and drop NA rows (same as notebook)
            train_df = self._create_lag_feature(train_df, target_column_name, lag=1)
            test_df = self._create_lag_feature(test_df, target_column_name, lag=1)

            # Drop rows with NA (first row will have NA after shift)
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)

            logging.info("Created time features and lag feature; dropped NA rows after lagging.")

            # Determine columns to use for modeling (exclude Date and target)
            exclude_cols = ["Date", target_column_name]
            # include the lag column as feature
            lag_col = f"prev_{target_column_name}"
            # If user provided lists, use them; otherwise auto-detect
            if numerical_columns is None or categorical_columns is None:
                auto_num, auto_cat = self._detect_column_types(train_df, exclude=exclude_cols)
                numerical_columns = numerical_columns or auto_num
                categorical_columns = categorical_columns or auto_cat

            # Ensure lag column is treated as numerical feature (add if not already)
            if lag_col not in numerical_columns and lag_col not in categorical_columns:
                numerical_columns = numerical_columns + [lag_col]

            # Remove any excluded columns if accidentally present
            numerical_columns = [c for c in numerical_columns if c not in exclude_cols]
            categorical_columns = [c for c in categorical_columns if c not in exclude_cols]

            logging.info(f"Final numerical columns: {numerical_columns}")
            logging.info(f"Final categorical columns: {categorical_columns}")

            preprocessor = self.get_data_transformer_object(numerical_columns, categorical_columns)

            # Separate inputs and targets
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]

            logging.info("Fitting preprocessor on training data.")
            X_train_arr = preprocessor.fit_transform(X_train)
            logging.info("Transforming test data.")
            X_test_arr = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]
            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocessor)
            logging.info(f"Saved preprocessor at: {self.config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
