# LLM-Driven Prediction of Blast-Induced Slope Failure (BISF)

## Project Overview
This project develops a prototype predictive model using LLM-based techniques for forecasting blast-induced slope failure during railway slope excavation in hard rock.

## Problem Context
- **Application**: Railway slope reconstruction using shallow jackhammer blasting
- **Blast Parameters**: 34mm diameter holes, ~1.5m deep
- **Risk**: Ground vibration damage, rockfalls, PPV safety violations
- **Goal**: Predict slope failure (Yes/No) and classify vibration risk

## Objectives
1. Reproduce baseline BISF prediction (LR, CART, RF models)
2. Design LLM-assisted workflow for preprocessing, feature engineering, rule extraction
3. Benchmark LLM-based/hybrid models against traditional approaches
4. Classify blasts into vibration risk classes

## Project Structure
```
Project2/
├── data/                      # Dataset storage
│   ├── raw/                   # Original blast database
│   └── processed/             # Cleaned and engineered features
├── models/                    # Trained model artifacts
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code
│   ├── data_preprocessing.py  # Data cleaning and preparation
│   ├── baseline_models.py     # Traditional ML models (LR, CART, RF)
│   ├── llm_feature_engineering.py  # LLM-driven feature creation
│   ├── llm_hybrid_model.py    # Hybrid LLM-ML pipeline
│   ├── evaluation.py          # Model evaluation metrics
│   └── utils.py               # Helper functions
├── config/                    # Configuration files
├── results/                   # Model outputs and reports
├── requirements.txt           # Python dependencies
└── main.py                    # Main execution script

```

## Input Features
- Burden (m)
- Spacing (m)
- Hole depth (m)
- Hole diameter (mm)
- Number of holes per round
- Maximum charge per delay (kg)
- Total charge (kg)
- Specific charge (kg/m³)
- Slope angle (degrees)
- Blast type
- Rock parameters (UCS, RQD, etc.)

## Output
- Binary classification: Slope Failure (Yes/No)
- Vibration risk class: Within limit / Exceeds limit

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Run complete pipeline
python main.py

# Run specific components
python src/baseline_models.py
python src/llm_hybrid_model.py
```

## Performance Metrics
- Accuracy
- Recall (Sensitivity)
- Specificity
- F-beta Score
- Confusion Matrix
- ROC-AUC

## Expected Outcomes
- Documented, reproducible toolchain
- Guidelines for LLM-enhanced BISF prediction
- Rapid risk assessment framework for field engineers
