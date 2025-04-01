import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DatetimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_cols: list[str], format: str | None = None):
        self.datetime_cols = datetime_cols
        self.format = format

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DatetimeTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert specified columns to datetime."""
        X = X.copy()
        for col in self.datetime_cols:
            X[col] = pd.to_datetime(X[col], format=self.format)
        return X


# Usage:
