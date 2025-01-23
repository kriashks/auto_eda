"""
AI-Augmented EDA with Optimized ML Integration
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from xgboost import XGBClassifier, XGBRegressor
from .eda_core import AutoEDA, EDAResults

class AIAugmentedEDA(AutoEDA):
    """Optimized ML-powered EDA with efficient memory usage"""
    
    def __init__(self, df: pd.DataFrame, target: Optional[str] = None):
        super().__init__(df, target)
        self.results = self._enhance_results()
    
    def _enhance_results(self) -> EDAResults:
        return EDAResults(
            **self.results.__dict__,
            feature_importance=self._get_feature_importance(),
            mutual_info_scores=self._get_mutual_info(),
            anomaly_scores=self._get_anomaly_scores()
        )
    
    def _get_feature_importance(self) -> Optional[pd.Series]:
        if not self.target:
            return None
            
        X = self.df.drop(columns=[self.target]).select_dtypes(include=np.number)
        y = self.df[self.target]
        
        model = XGBClassifier() if y.dtype == 'object' else XGBRegressor()
        model.fit(X, y)
        return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    def _get_mutual_info(self) -> Optional[pd.Series]:
        if not self.target:
            return None
            
        X = self.df.drop(columns=[self.target]).select_dtypes(include=np.number)
        y = self.df[self.target]
        
        mi = mutual_info_classif(X, y) if y.dtype == 'object' else mutual_info_regression(X, y)
        return pd.Series(mi, index=X.columns).sort_values(ascending=False)
    
    def _get_anomaly_scores(self) -> pd.Series:
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            clf = IsolationForest(n_estimators=100)
            return pd.Series(clf.fit_predict(self.df[numeric_cols]), name='anomaly_score')
        return pd.Series()
