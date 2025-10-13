# ml-project-credit-risk-model

# Credit Risk Classification Model

A sophisticated machine learning solution for predicting credit default risk using advanced classification techniques, feature engineering, and class imbalance handling strategies to maximize default detection while maintaining high overall accuracy.

## 📋 Project Overview

This project implements a comprehensive credit risk assessment system that predicts the likelihood of loan default. By employing Weight of Evidence (WoE) analysis, Information Value (IV) feature selection, and multiple resampling techniques, the solution achieves **95% recall** for default class detection with an impressive **AUC of 98.37%**.

## 🎯 Problem Statement

Credit risk assessment is critical for financial institutions to minimize losses and make informed lending decisions. The primary challenges include:

- **Class Imbalance**: Default cases are typically rare compared to non-default cases
- **False Negatives Cost**: Missing a default prediction is more costly than false alarms
- **Feature Complexity**: Multiple correlated features requiring careful selection
- **Model Interpretability**: Need for explainable predictions for regulatory compliance

This project addresses these challenges by prioritizing recall (sensitivity) to ensure maximum detection of potential defaults while maintaining overall model accuracy.

## 🔍 Key Features

- **Advanced Feature Engineering**: WoE and IV-based feature selection (IV > 0.02)
- **Multi-Algorithm Comparison**: Logistic Regression, Random Forest, and XGBoost
- **Class Imbalance Handling**: RandomUnderSampler and SMOTE-Tomek techniques
- **Comprehensive Optimization**: RandomizedSearchCV, Optuna, and Bayesian Optimization
- **Rigorous Evaluation**: ROC-AUC, KS Statistics, Gini Coefficient, and Rank Ordering
- **Production-Ready Deployment**: Focus on high recall for default detection

## 🚀 Methodology

### 1. Data Preparation
- Loaded and merged multiple datasets for comprehensive credit profiles
- Performed train-test split excluding target variable ('Default')
- Handled missing values and removed duplicate records
- Removed unique identifier columns with no predictive power

### 2. Exploratory Data Analysis
- **Numerical Features**: Used boxplots and histograms for outlier detection
- **Categorical Features**: Analyzed and cleaned sub-categories for consistency
- **Target Relationship**: Visualized correlations between features and default status
- **Correlation Analysis**: Used heatmaps to detect multicollinearity

### 3. Feature Engineering & Selection

#### Statistical Feature Selection
- **Weight of Evidence (WoE)**: Calculated for all features to measure predictive power
- **Information Value (IV)**: Selected features with IV > 0.02
  - IV < 0.02: Not useful for prediction
  - 0.02 ≤ IV < 0.1: Weak predictive power
  - 0.1 ≤ IV < 0.3: Medium predictive power
  - IV ≥ 0.3: Strong predictive power

#### Feature Enhancement
- Created new features to capture complex relationships
- Calculated VIF scores to detect and remove multicollinear features
- Applied same transformations to test set for consistency
- Encoded categorical variables for model compatibility

### 4. Model Development & Evolution

#### Phase 1: Baseline Models (Without Class Imbalance Handling)
| Model | Accuracy | Recall (Default) | Issues |
|-------|----------|------------------|--------|
| Logistic Regression | 96% | 72% | Low default detection |
| Random Forest | 97% | 72% | Low default detection |
| XGBoost | 96% | 75% | Low default detection |

**Key Insight**: High accuracy but insufficient recall for critical default class.

#### Phase 2: Hyperparameter Optimization (No Resampling)
| Model | Accuracy | Recall (Default) | Optimization |
|-------|----------|------------------|--------------|
| Logistic Regression | 96% | 74% | RandomizedSearchCV |
| XGBoost | 96% | 84% | RandomizedSearchCV |

#### Phase 3: RandomUnderSampler
| Model | Accuracy | Recall (Default) | Precision (Default) |
|-------|----------|------------------|---------------------|
| Logistic Regression | 92% | 95% | 51% |
| XGBoost | 92% | 98% | 53% |

**Trade-off**: Significantly improved recall but with reduced precision.

#### Phase 4: SMOTE-Tomek (Balanced Approach)

**Logistic Regression + Optuna:**
- Recall: **95%**
- Accuracy: **93%**
- Precision: **56%**

**XGBoost + Optuna:**
- Recall: **88%**
- Accuracy: **96%**
- Precision: **71%**

#### Phase 5: Final Production Model

**Logistic Regression + SMOTE-Tomek + Bayesian Optimization:**
- **Recall: 95%** ✅
- **Accuracy: 93%** ✅
- **Precision: 56%** ✅
- **AUC: 98.37%** ✅
- **KS Statistic: 85.98%** (at Decile 8) ✅
- **Gini Coefficient: 0.9676** ✅

**Selected for deployment due to optimal balance of high recall and acceptable precision.**

## 📊 Model Evaluation Metrics

### ROC-AUC Analysis
- **AUC Score**: 98.37% - Excellent discrimination capability
- Indicates the model can effectively distinguish between default and non-default cases

### Kolmogorov-Smirnov (KS) Statistic
- **KS Value**: 85.98% at Decile 8
- Measures the maximum separation between cumulative distributions
- High KS indicates strong predictive power

### Gini Coefficient
- **Gini**: 0.9676
- Measures inequality in model predictions
- Value close to 1 indicates excellent model performance

### Rank Ordering
- Analyzed model performance across risk deciles
- Confirmed consistent discrimination across score ranges

## 🛠️ Technologies Used

- **Programming Language**: Python 3.8+
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Class Imbalance**: imbalanced-learn (SMOTE, Tomek Links)
- **Hyperparameter Optimization**: Optuna, scikit-optimize (Bayesian Optimization)
- **Statistical Analysis**: SciPy, Statsmodels

## 📁 Project Structure

```
credit-risk-classification/
│
├── data/
│   ├── raw/                    # Original datasets
│   ├── merged/                 # Merged datasets
│   ├── processed/              # Cleaned and preprocessed data
│   └── features/               # Engineered features with WoE/IV
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_imbalance_handling.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   └── 07_model_evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py   # Data cleaning and merging
│   ├── feature_engineering.py  # WoE, IV calculations
│   ├── model_training.py       # Model training pipeline
│   ├── evaluation_metrics.py   # KS, Gini, ROC calculations
│   └── prediction.py           # Prediction interface
│
├── models/
│   ├── logistic_smote_bayesian.pkl  # Production model
│   └── model_metadata.json     # Model parameters and metrics
│
├── config/
│   └── model_config.yaml       # Configuration parameters
│
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-risk-classification.git
cd credit-risk-classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model

```python
from src.model_training import train_credit_model

# Train the model with SMOTE-Tomek and Bayesian Optimization
model, metrics = train_credit_model(
    data_path='data/processed/credit_data.csv',
    use_smote=True,
    optimization='bayesian'
)

print(f"Model Recall: {metrics['recall']:.2%}")
print(f"Model AUC: {metrics['auc']:.2%}")
```

### Making Predictions

```python
from src.prediction import predict_default_risk

# Predict default probability for a new applicant
applicant_data = {
    'income': 50000,
    'debt_to_income': 0.35,
    'credit_history_length': 5,
    'payment_history': 'good',
    'employment_status': 'employed'
}

risk_score, risk_category = predict_default_risk(applicant_data)
print(f"Default Probability: {risk_score:.2%}")
print(f"Risk Category: {risk_category}")
```

### Model Evaluation

```python
from src.evaluation_metrics import calculate_credit_metrics

# Calculate comprehensive evaluation metrics
metrics = calculate_credit_metrics(y_true, y_pred, y_proba)

print(f"KS Statistic: {metrics['ks_stat']:.2%}")
print(f"Gini Coefficient: {metrics['gini']:.4f}")
print(f"AUC: {metrics['auc']:.2%}")
```

## 📈 Feature Importance

Top predictive features based on Information Value:

1. **Payment History** (IV: 0.45) - Strong predictor
2. **Debt-to-Income Ratio** (IV: 0.38) - Strong predictor
3. **Credit Utilization** (IV: 0.32) - Strong predictor
4. **Employment Status** (IV: 0.28) - Medium predictor
5. **Credit History Length** (IV: 0.22) - Medium predictor

## 🎓 Key Learnings

### Class Imbalance Strategy
- **SMOTE-Tomek** provided the best balance between recall and precision
- **RandomUnderSampler** achieved highest recall but with significant precision loss
- Hybrid approach (SMOTE-Tomek) creates synthetic samples while removing borderline cases

### Optimization Techniques
- **Bayesian Optimization** outperformed RandomizedSearchCV for this use case
- **Optuna** provided efficient hyperparameter search with pruning
- Multiple optimization runs helped avoid local minima

### Business Impact
- **95% recall** ensures detection of most potential defaults
- **56% precision** means approximately 44% false positives - acceptable for risk-averse lending
- High AUC (98.37%) indicates excellent overall discrimination capability

## 🏦 Business Applications

### Risk-Based Pricing
Use predicted probabilities to set appropriate interest rates based on risk levels.

### Automated Decision Making
- **Low Risk** (0-20% probability): Auto-approve with standard terms
- **Medium Risk** (20-60% probability): Manual review required
- **High Risk** (60-100% probability): Auto-reject or require additional collateral

### Portfolio Management
Monitor portfolio health by tracking distribution across risk deciles.

## ⚠️ Model Limitations

1. **Precision Trade-off**: 56% precision means manual review of flagged applications needed
2. **Data Dependency**: Model performance depends on feature availability
3. **Temporal Drift**: Economic conditions may affect model performance over time
4. **Interpretability**: SMOTE generates synthetic data which may not represent real scenarios

## 🔮 Future Enhancements

- [ ] Implement ensemble methods combining multiple algorithms
- [ ] Add explainability features using SHAP values
- [ ] Develop real-time scoring API
- [ ] Implement model monitoring and drift detection
- [ ] Add more granular risk segmentation
- [ ] Incorporate external data sources (macroeconomic indicators)
- [ ] A/B testing framework for model updates
- [ ] Fairness and bias analysis across demographic groups

## 📊 Model Monitoring

### Key Metrics to Track
- **Population Stability Index (PSI)**: Monitor feature distribution drift
- **Characteristic Stability Index (CSI)**: Track score distribution changes
- **Approval Rate**: Ensure consistency with business objectives
- **Default Rate**: Validate model predictions against actual outcomes



## 📚 References

1. Weight of Evidence and Information Value in credit scoring
2. SMOTE: Synthetic Minority Over-sampling Technique
3. Bayesian Optimization for hyperparameter tuning
4. Kolmogorov-Smirnov test in credit risk models
5. Regulatory guidelines for credit risk modeling

---

**Disclaimer**: This model is intended for educational and demonstration purposes. Always consult with domain experts and ensure compliance with relevant regulations before deploying credit risk models in production environments.
