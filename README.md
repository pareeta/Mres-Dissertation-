# Fairness-Aware Neural Networks for Stroke and Heart Disease Prediction

A Research dissertation project implementing bias mitigation techniques in custom neural networks for clinical risk prediction, with fairness evaluation across age and gender groups.

## Overview

This project develops and evaluates fairness-aware deep learning models for predicting stroke and heart disease. Three bias mitigation strategies are compared: pre-processing (reweighing), in-processing (adversarial debiasing), and post-processing (threshold optimisation). Fairness is measured using Equal Opportunity Difference (EOD) across age groups.

## Datasets

- **Stroke dataset**: [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) — 5,110 patients, 249 stroke cases (Kaggle, fedesoriano)
- **Heart disease dataset**: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) — 918 patients, 508 with heart disease (Kaggle, fedesoriano)

## Methods

### Custom Neural Network Architecture
- Input layer (input_dim features)
- Hidden layer 1: Linear(64) + BatchNorm + ReLU + Dropout(0.3)
- Hidden layer 2: Linear(32) + ReLU + Dropout(0.2)
- Hidden layer 3: Linear(16) + ReLU
- Output layer: Linear(1) + Sigmoid
- Loss: Focal Loss (alpha=0.25, gamma=2)

### Pre-processing Pipeline
- Stratified 80/20 train-test split
- KNN imputation for BMI (fit on train only)
- StandardScaler normalisation (fit on train only)
- SMOTE oversampling for stroke dataset (train only)

### Bias Mitigation Techniques
1. **Reweighing** (pre-processing): AIF360 sample weights applied with per-sample Focal Loss
2. **Adversarial Debiasing** (in-processing): Custom adversarial NN with 3-class age adversary (lambda=1.0)
3. **Threshold Optimisation** (post-processing): Fairlearn ThresholdOptimizer with equalised odds constraint on Logistic Regression

### Protected Attributes
- Gender (binary)
- Age group: under 40, 40-60, over 60



### Key Finding
SMOTE worsens age fairness in stroke prediction because synthetic samples are generated predominantly from older patients who make up the majority of stroke cases.



## Requirements

```
torch
numpy
pandas
scikit-learn
imbalanced-learn
aif360
fairlearn
shap
matplotlib
```

Install all dependencies:

```bash
pip install torch numpy pandas scikit-learn imbalanced-learn aif360 fairlearn shap matplotlib
```

## Usage

Open `FairnessandBiasmitigation.ipynb` in Jupyter Notebook or Google Colab and run all cells in order.

The notebook is structured as follows:
1. Data loading and preprocessing
2. Baseline neural network training and evaluation
3. Benchmark models (Logistic Regression, Random Forest, XGBoost)
4. Fairness audit (baseline)
5. Bias mitigation: reweighing
6. Bias mitigation: adversarial debiasing
7. Bias mitigation: threshold optimisation
8. SHAP analysis
9. Results comparison


