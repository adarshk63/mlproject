import sys
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
  
    def __init__(self, model_path: str = None, preprocessor_path: str = None, target_column_name: str = "Units Sold"):
        self.model_path = model_path or os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = preprocessor_path or os.path.join("artifacts", "preprocessor.pkl")
        self.target_column_name = target_column_name

    def _ensure_date_and_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                if df["Date"].isna().any():
                    dates = pd.date_range(start="2020-01-01", periods=len(df), freq="D")
                    df["Date"] = df["Date"].fillna(pd.Series(dates, index=df.index))
            else:
                df["Date"] = pd.date_range(start="2020-01-01", periods=len(df), freq="D")

            df["year"] = df["Date"].dt.year
            df["month"] = df["Date"].dt.month
            df["day"] = df["Date"].dt.day
            df["weekday"] = df["Date"].dt.weekday
            df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
            df["month_start"] = df["Date"].dt.is_month_start.astype(int)
            df["month_end"] = df["Date"].dt.is_month_end.astype(int)

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: Union[pd.DataFrame, Dict[str, Any]]):
        try:
            # load artifacts
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            # convert dict -> DataFrame if needed
            if isinstance(features, dict):
                input_df = pd.DataFrame([features])
            elif isinstance(features, list):
                # list of dicts
                input_df = pd.DataFrame(features)
            elif isinstance(features, pd.DataFrame):
                input_df = features.copy()
            else:
                raise ValueError("Unsupported features type. Provide pandas.DataFrame or dict/list of dicts.")

            # Create date/time features to match training preprocessing
            input_df = self._ensure_date_and_time_features(input_df)

            # Check presence of the required lag column for single-row prediction:
            lag_col = f"prev_{self.target_column_name}"
            if lag_col not in input_df.columns:
                    raise ValueError(
                        f"Missing required lag feature '{lag_col}' for single-row prediction. "
                        f"Provide previous period value in this column."
                    )
            if self.target_column_name in input_df.columns:
                input_df = input_df.drop(columns=[self.target_column_name])

            # Transform features with saved preprocessor
            X_transformed = preprocessor.transform(input_df)

            # Predict
            preds = model.predict(X_transformed)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


@dataclass
class CustomData:
   
    prev_target_value: float
    date: Optional[str] = None
    extra_features: Optional[Dict[str, Any]] = None
    target_column_name: str = "Units Sold"

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            row = {}

            # Date handling
            if self.date is not None:
                row["Date"] = pd.to_datetime(self.date)
            else:
                # If no date supplied, use today's date (or you can set a default)
                row["Date"] = pd.Timestamp.now().normalize()

            # Lag column (required)
            lag_col = f"prev_{self.target_column_name}"
            row[lag_col] = self.prev_target_value

            # Add any extra features user supplied
            if self.extra_features:
                for k, v in self.extra_features.items():
                    row[k] = v

            df = pd.DataFrame([row])
            return df

        except Exception as e:
            raise CustomException(e, sys)
