"""
Core EDA Functionality - Optimized with Vectorization
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class EDAResults:
    """Container for EDA results using memory-efficient types"""
    shape: tuple
    dtypes: 'pd.Series[object]'
    numeric_stats: pd.DataFrame
    categorical_stats: pd.DataFrame
    missing_values: 'pd.Series[int]'
    duplicate_rows: int
    correlation_matrix: pd.DataFrame
    target_correlations: Optional[pd.Series] = None
    outlier_info: Optional[Dict[str, int]] = None
    time_series_analysis: Optional[Dict[str, Any]] = None

class AutoEDA:
    """Vectorized EDA implementation for performance"""
    
    def __init__(self, df: pd.DataFrame, target: Optional[str] = None):
        self.df = df.copy()
        self.target = target
        self.results = self._perform_eda()
    
    def _perform_eda(self) -> EDAResults:
        """Optimized EDA computation using vectorization"""
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        cat_cols = self.df.select_dtypes(include='object').columns
        
        return EDAResults(
            shape=self.df.shape,
            dtypes=self.df.dtypes,
            numeric_stats=self._numeric_stats(numeric_cols),
            categorical_stats=self._categorical_stats(cat_cols),
            missing_values=self.df.isna().sum(),
            duplicate_rows=self.df.duplicated().sum(),
            correlation_matrix=self.df[numeric_cols].corr(),
            target_correlations=self._target_correlations(numeric_cols),
            outlier_info=self._detect_outliers(numeric_cols),
            time_series_analysis=self._time_series_checks()
        )
    
    def _numeric_stats(self, cols) -> pd.DataFrame:
        return self.df[cols].describe(percentiles=[.25, .5, .75, .95, .99]).T
    
    def _categorical_stats(self, cols) -> pd.DataFrame:
        if not cols.empty:
            return pd.concat([
                self.df[cols].nunique().rename('unique'),
                self.df[cols].mode().iloc[0].rename('mode'),
                self.df[cols].apply(lambda x: x.value_counts(normalize=True).iloc[0]).rename('freq')
            ], axis=1)
        return pd.DataFrame()
    
    def _target_correlations(self, numeric_cols):
        if self.target and self.target in numeric_cols:
            return self.df[numeric_cols].corr()[self.target].sort_values(ascending=False)
        return None
    
    def _detect_outliers(self, numeric_cols, threshold: float = 3.0):
        if not numeric_cols.empty:
            z = np.abs((self.df[numeric_cols] - self.df[numeric_cols].mean()) / self.df[numeric_cols].std())
            return (z > threshold).sum().to_dict()
        return {}
    
    def _time_series_checks(self):
        time_cols = self.df.columns[self.df.columns.str.contains('date|time', case=False)]
        if not time_cols.empty:
            return {
                col: {
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'n_unique': self.df[col].nunique()
                } for col in time_cols
            }
        return None
