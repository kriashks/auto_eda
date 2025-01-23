"""
Optimized Utilities for EDA Pipeline
"""

from typing import Dict, Any
import pandas as pd

def format_report(results: Dict[str, Any]) -> str:
    """Vectorized report formatting"""
    report = []
    for section, content in results.items():
        if isinstance(content, pd.DataFrame):
            report.append(f"{section}:\n{content.to_markdown()}")
        elif isinstance(content, dict):
            report.append(f"{section}:\n" + "\n".join(f"{k}: {v}" for k,v in content.items()))
        else:
            report.append(f"{section}: {content}")
    return "\n\n".join(report)

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage"""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'object':
            df[col] = df[col].astype('category')
        elif np.issubdtype(col_type, np.integer):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        else:
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df
