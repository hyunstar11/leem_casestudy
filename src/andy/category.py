import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from andy.prep import bin_job_titles, normalize_title


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols: list[str], cutoff: int = 35) -> None:
        self.categorical_cols = categorical_cols
        self.cutoff = cutoff
        self.low_card_cols: list[str] = []

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "CategoricalTransformer":
        self.low_card_cols = [
            col for col in self.categorical_cols if X[col].nunique() <= self.cutoff
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.low_card_cols:
            X[col] = X[col].astype("category")
        return X


class OrdinalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_mappings: dict[str, list[str]]) -> None:
        """
        Parameters:
            ordinal_mappings: dict mapping column names to their ordered categories
            e.g. {
                'grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                'emp_length': ['< 1 year', '1 year', '2 years', ...],
                'home_ownership': ['OWN', 'MORTGAGE', 'RENT', 'OTHER']
            }
        """
        self.ordinal_mappings = ordinal_mappings

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OrdinalTransformer":
        # Validate all mapped columns exist
        missing_cols = set(self.ordinal_mappings) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col, categories in self.ordinal_mappings.items():
            X[col] = pd.Categorical(X[col], categories=categories, ordered=True)
        return X


class AddressTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, address_col: str = "address", drop_original: bool = True
    ) -> None:
        self.address_col = address_col
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "AddressTransformer":
        if self.address_col not in X.columns:
            raise ValueError(f"Address column '{self.address_col}' not found")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Extract features from address
        X["military"] = (
            X[self.address_col]
            .str.split(" ")
            .str[0]
            .str.startswith("US")
            .astype("category")
        )

        X["state"] = (
            X[self.address_col]
            .str.split("\r\n")
            .str[-1]
            .str.split(" ")
            .str[-2]
            .astype("category")
        )

        # Drop original address column if requested
        if self.drop_original:
            X = X.drop(columns=[self.address_col])

        return X


class JobTitleTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        emp_title_col: str = "emp_title",
        job_map: dict[str, str] | None = None,
        n_common_titles: int = 30,
        similarity_threshold: float = 0.6,
        drop_intermediate: bool = True,
    ):
        self.emp_title_col: str = emp_title_col
        self.job_map: dict[str, str] = job_map or {}
        self.n_common_titles: int = n_common_titles
        self.similarity_threshold: float = similarity_threshold
        self.drop_intermediate: bool = drop_intermediate
        self.common_job_titles: list[str] = []
        self.title_to_category: dict[str, str] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "JobTitleTransformer":
        # Normalize titles and extract last word
        emp_titles = X[self.emp_title_col].apply(normalize_title)
        job_titles = emp_titles.str.split().str[-1]

        # Apply manual mapping
        job_titles = job_titles.apply(lambda x: self.job_map.get(x, x))

        # Get most common titles
        self.common_job_titles = (
            job_titles.value_counts().head(self.n_common_titles).index.tolist()
        )

        # Create groups using bin_job_titles
        groups = bin_job_titles(
            self.common_job_titles, job_titles, threshold=self.similarity_threshold
        )

        # Create reverse mapping
        self.title_to_category = {}
        for category, titles in groups.items():
            for title in titles:
                self.title_to_category[title] = category

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Normalize and extract job titles
        X["emp_title_norm"] = X[self.emp_title_col].apply(normalize_title)
        X["job_title"] = X["emp_title_norm"].str.split().str[-1]

        # Apply manual mapping
        X["job_title"] = X["job_title"].apply(lambda x: self.job_map.get(x, x))

        # Use common titles where possible, otherwise use normalized emp_title
        X["job_title"] = X.apply(
            lambda row: (
                row.job_title
                if row.job_title in self.common_job_titles
                else row.emp_title_norm
            ),
            axis=1,
        )

        # Map to categories
        X["job"] = (
            X["job_title"]
            .apply(lambda x: self.title_to_category.get(x, "other"))
            .astype("category")
        )

        # Drop intermediate columns if requested
        if self.drop_intermediate:
            X = X.drop(columns=[self.emp_title_col, "emp_title_norm", "job_title"])

        return X


class PurposeTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        purpose_col: str = "purpose",
        title_col: str = "title",
        drop_title: bool = True,
    ):
        self.purpose_col = purpose_col
        self.title_col: str = title_col
        self.drop_title: bool = drop_title
        self.purpose_categories: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PurposeTransformer":
        # Store unique purposes for consistent categories
        self.purpose_categories = sorted(X[self.purpose_col].unique())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Convert purpose to categorical with consistent categories
        X[self.purpose_col] = pd.Categorical(
            X[self.purpose_col], categories=self.purpose_categories
        )

        # Drop redundant title column if requested
        if self.drop_title and self.title_col in X.columns:
            X = X.drop(columns=[self.title_col])

        return X