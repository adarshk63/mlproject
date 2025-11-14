import sys
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """
    Prediction pipeline for time-series / demand-forecasting models.
    - Expects artifacts/model.pkl and artifacts/preprocessor.pkl to exist.
    - Input can be a pandas.DataFrame or output of CustomData.get_data_as_data_frame().
    - The dataframe must include the lag column named 'prev_<target_name>' (e.g. 'prev_Demand Forecast')
      when predicting a single/future row.
    """

    def __init__(self, model_path: str = None, preprocessor_path: str = None, target_column_name: str = "Demand Forecast"):
        self.model_path = model_path or os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = preprocessor_path or os.path.join("artifacts", "preprocessor.pkl")
        self.target_column_name = target_column_name

    def _ensure_date_and_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure Date column is parsed and add the same time features as used in training.
        """
        try:
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                # fill invalid dates with a synthetic range if any are NaT
                if df["Date"].isna().any():
                    dates = pd.date_range(start="2020-01-01", periods=len(df), freq="D")
                    df["Date"] = df["Date"].fillna(pd.Series(dates, index=df.index))
            else:
                # if Date missing, create a synthetic date range (not ideal for real predictions)
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
        """
        features: either a pandas DataFrame with the same columns used for training (excluding target),
                  or a dict representing a single row (or list of dicts).
        Returns model predictions (numpy array or list).
        """
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
                # If user passed multiple rows and preprocessor was fit on other columns, they might not need prev_*
                # but for single-row forecasting the lag value must be provided externally.
                if len(input_df) == 1:
                    raise ValueError(
                        f"Missing required lag feature '{lag_col}' for single-row prediction. "
                        f"Provide previous period value in this column."
                    )
                # otherwise continue — maybe batch input already has required features

            # Drop target column if accidentally present
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
    """
    Helper to prepare a single prediction row for the demand-forecasting model.

    Required:
      - prev_target_value: the previous period's true value (float). This will be placed in column
        'prev_<target_column_name>' which the preprocessor expects.
    Optional:
      - date: string or pd.Timestamp for the row to predict (used to derive time features).
      - extra_features: dict of other feature_name: value pairs that the model expects (e.g., categorical flags).
    """

    prev_target_value: float
    date: Optional[str] = None
    extra_features: Optional[Dict[str, Any]] = None
    target_column_name: str = "Demand Forecast"

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

            # Note: time features are created inside PredictPipeline._ensure_date_and_time_features,
            # but creating them here is also fine. Keep pipeline consistent by doing it in PredictPipeline.
            return df

        except Exception as e:
            raise CustomException(e, sys)
