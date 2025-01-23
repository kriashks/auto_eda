# AutoEDA: Automated Exploratory Data Analysis Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-layered EDA toolkit for technical interviews and data analysis, combining traditional statistics, ML-powered insights, and LLM-augmented hypothesis generation.

## Features

### Core EDA
- Basic statistics & distributions
- Missing value analysis
- Correlation matrices
- Outlier detection
- Time series analysis

### AI-Augmented
- Feature importance (XGBoost)
- Mutual information scoring
- Anomaly detection (Isolation Forest)
- Automated feature suggestions

### LLM-Enhanced
- Interview question generation
- Hypothesis suggestions
- Business use case proposals
- Self-improving analysis (via AI critic)
- Model-swappable architecture

### Performance
- Vectorized operations
- Memory optimization
- Model caching
- 8-bit LLM quantization

## Installation

```bash
pip install pandas numpy scikit-learn xgboost transformers torch
