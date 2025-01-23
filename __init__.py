"""
auto_eda - Automated Exploratory Data Analysis Toolkit
"""

__version__ = "1.0.0"

# Core exports
from .eda_core import AutoEDA, EDAResults
from .ai_augmented import AIAugmentedEDA
from .llm_augmented import LLMAugmentedEDA
from .utils import optimize_memory

# Top-level API
from .llm_augmented import analyze_data  # Main entry point

# Optional: Configure package-level logging
import logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

__all__ = [
    'AutoEDA',
    'AIAugmentedEDA',
    'LLMAugmentedEDA',
    'EDAResults',
    'analyze_data',
    'optimize_memory'
]
