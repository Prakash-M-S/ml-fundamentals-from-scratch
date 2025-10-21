# Logistic Regression: Pulsar Detection 🌟

## Overview

Implementation of **Logistic Regression from scratch** following Andrew Ng's Machine Learning Course concepts, applied to binary classification of pulsar candidates. This project demonstrates key classification algorithms including sigmoid function, cross-entropy loss, and regularized gradient descent.

## Dataset

- **Source**: Pulsar candidate data from radio telescope observations
- **Features**: 8 statistical measures of pulsar signal characteristics
- **Target**: Binary classification (1 = Pulsar, 0 = Non-pulsar)
- **Challenge**: Imbalanced dataset requiring careful evaluation

## Key Implementations

### 🔧 From Scratch Implementation

- **Sigmoid Function**: Logistic activation for probability mapping
- **Cost Function**: Cross-entropy loss with L2 regularization
- **Gradient Descent**: Custom implementation with α = 0.5, λ = 1
- **Regularization**: L2 penalty to prevent overfitting
- **Stratified Split**: Maintains class balance in train/test sets

### 📊 Scikit-learn Comparison

Direct performance comparison between custom implementation and sklearn's LogisticRegression.

## Results

| Metric | From Scratch | Scikit-learn |
|--------|-------------|--------------|
| Accuracy | 98.16% | 98.16% |
| Precision | 96.48% | 96.48% |
| Recall | 83.12% | 83.12% |
| F1-Score | 89.30% | 89.30% |

*Perfect agreement between implementations demonstrates the correctness of the custom logistic regression with L2 regularization.*

## Visualizations

- **Sigmoid Curve**: Activation function visualization
- **Decision Boundary**: Classification threshold demonstration
- **Confusion Matrix**: Detailed classification performance
- **Learning Curve**: Cost function convergence

## Key Learnings from Andrew Ng's Course

1. **Sigmoid Function**: Maps any real number to (0,1) probability range
2. **Cross-entropy Loss**: Appropriate cost function for classification
3. **Regularization**: L2 penalty prevents overfitting in high-dimensional data
4. **Classification Metrics**: Beyond accuracy - precision, recall, F1-score
5. **Imbalanced Data**: Stratified sampling and appropriate metrics

## How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Execution

1. Open `notebooks/scratch.ipynb` in Jupyter/VS Code
2. Run all cells sequentially
3. View results in `results/` folder

## File Structure

```text
logistic_regression_pulsar/
├── README.md
├── data/
│   └── pulsar_data_train.csv    # Pulsar candidate dataset
├── notebooks/
│   └── scratch.ipynb            # Main implementation notebook
└── results/
    ├── sigmoid_curve.png        # Sigmoid function plot
    ├── decision_boundary.png    # Classification boundary
    └── confusion_matrix.png     # Performance visualization
```

## Andrew Ng Course Concepts Demonstrated

- ✅ **Logistic Regression Theory**: Mathematical foundations
- ✅ **Sigmoid Function**: Probability mapping and interpretation
- ✅ **Cross-entropy Cost**: Appropriate loss for classification
- ✅ **Regularization**: L2 penalty implementation
- ✅ **Gradient Descent**: Parameter optimization for classification
- ✅ **Classification Metrics**: Comprehensive performance evaluation

## Technical Highlights

- **Numerical Stability**: Added epsilon (1e-15) to prevent log(0)
- **Missing Value Handling**: Mean imputation for robust preprocessing
- **Feature Scaling**: Z-score normalization for gradient descent stability
- **Regularization Parameter**: λ = 1 for balanced bias-variance tradeoff

## Advanced Features Implemented

- ✅ **L2 Regularization**: Ridge penalty for overfitting prevention (λ = 1)
- ✅ **Stratified Sampling**: Maintains class balance in imbalanced dataset
- ✅ **Numerical Stability**: Epsilon handling for log(0) prevention
- ✅ **Comprehensive Metrics**: Precision, Recall, F1-score for classification
- ✅ **Feature Scaling**: Z-score normalization for stability

## Next Steps

- [ ] ROC curve and AUC analysis
- [ ] Cross-validation for hyperparameter tuning
- [ ] Advanced optimization algorithms (Adam, RMSprop)
- [ ] Elastic Net regularization (L1 + L2 combination)

---

*This project demonstrates practical application of Andrew Ng's Machine Learning Course classification concepts. The from-scratch implementation provides deep understanding of logistic regression mechanics before using high-level libraries.*
