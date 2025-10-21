# Machine Learning Fundamentals from Scratch 🤖

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **From Theory to Practice**: Complete implementations of fundamental machine learning algorithms following Andrew Ng's Machine Learning Specialization.

## 🎯 Project Overview

This repository contains **from-scratch implementations** of core machine learning algorithms, built to understand the mathematical foundations before using high-level libraries. Each implementation is validated against scikit-learn for correctness.

### 🚀 What's Inside

| Algorithm | Problem Type | Dataset | Key Features |
|-----------|-------------|---------|--------------|
| **Linear Regression** | Regression | Bike Sharing | Gradient Descent, Ridge Regularization, Cross-Validation |
| **Logistic Regression** | Classification | Pulsar Detection | Sigmoid, Cross-Entropy, L2 Regularization, Stratified Sampling |

## 📊 Results Summary

### Linear Regression - Bike Rental Prediction
- **MAE**: 651.99 (vs sklearn: 651.96) - 99.99% agreement
- **RMSE**: 859.36 rentals
- **Cross-Validation**: 5-fold CV with low variance (2.8% coefficient of variation)
- **Features**: 9 weather and temporal features predicting daily bike rentals

### Logistic Regression - Pulsar Detection
- **Accuracy**: 98.16% 
- **Precision**: 96.48%
- **Recall**: 83.12%
- **F1-Score**: 89.30%
- **Perfect sklearn validation** across all metrics

## 🛠️ Technical Implementations

### Core Algorithms
- ✅ **Custom Gradient Descent** with learning rate optimization
- ✅ **Feature Scaling** (Z-score normalization)
- ✅ **Regularization** (L2/Ridge for both algorithms)
- ✅ **Cross-Validation** for robust evaluation
- ✅ **Stratified Sampling** for imbalanced datasets

### Advanced Features
- ✅ **Overfitting Analysis** with comprehensive diagnostics
- ✅ **Learning Curve Visualization** 
- ✅ **Numerical Stability** (epsilon handling)
- ✅ **Multiple Evaluation Metrics**
- ✅ **Scikit-learn Validation** for correctness verification

## 📁 Repository Structure

```
ml-regression-projects/
├── linear_regression_bikes/
│   ├── data/day.csv                    # Bike sharing dataset
│   ├── notebooks/scratch.ipynb         # Complete implementation
│   ├── results/                        # Visualizations and plots
│   └── README.md                       # Detailed project docs
├── logistic_regression_pulsar/
│   ├── data/pulsar_data_train.csv      # Pulsar detection dataset
│   ├── notebooks/scratch.ipynb         # Complete implementation  
│   ├── results/                        # Visualizations and plots
│   └── README.md                       # Detailed project docs
├── requirements.txt                     # Dependencies
└── README.md                           # This file
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Projects
```bash
# Linear Regression
cd linear_regression_bikes/notebooks/
jupyter notebook scratch.ipynb

# Logistic Regression  
cd logistic_regression_pulsar/notebooks/
jupyter notebook scratch.ipynb
```

## 🎓 Learning Outcomes

### Mathematical Foundations
- **Gradient Descent**: Understanding parameter optimization mechanics
- **Cost Functions**: MSE for regression, Cross-entropy for classification
- **Regularization**: Bias-variance tradeoff and overfitting prevention
- **Feature Engineering**: Scaling, normalization, and preprocessing

### Implementation Skills
- **NumPy Vectorization**: Efficient mathematical operations
- **Algorithm Design**: Clean, modular function architecture
- **Validation Methodology**: Train/test splits, cross-validation, metrics
- **Debugging**: Mathematical correctness verification

### Professional Practices
- **Documentation**: Clear README files with results and methodology
- **Reproducibility**: Seeded random states and consistent preprocessing
- **Visualization**: Learning curves, performance plots, diagnostic charts
- **Version Control**: Clean Git history with meaningful commits

## 🔍 Validation & Quality Assurance

- ✅ **Perfect Sklearn Agreement**: All implementations match sklearn results
- ✅ **Cross-Validation Stability**: Low variance across different data splits
- ✅ **Overfitting Analysis**: Comprehensive diagnostics confirm model generalization
- ✅ **Code Quality**: Clean, documented, modular implementations

## 🌟 Key Insights

> *"Sometimes the best way to understand a model is to build it from nothing."*

1. **Gradient Descent Intuition**: Watching cost functions converge builds deep understanding
2. **Regularization Necessity**: L2 penalty prevents overfitting in practice
3. **Scaling Criticality**: Feature normalization dramatically improves convergence
4. **Validation Importance**: Multiple evaluation methods catch edge cases

## 📈 What's Next

- [ ] Neural Networks from scratch
- [ ] Support Vector Machines
- [ ] Decision Trees and Random Forests
- [ ] Advanced optimization algorithms (Adam, RMSprop)

## 🙏 Acknowledgments

Built following **Andrew Ng's Machine Learning Specialization** (Supervised ML course). The mathematical foundations and best practices learned from the course made these implementations possible.

---

### 📫 Connect & Discuss

- **LinkedIn**: [Prakash M S](https://www.linkedin.com/in/prakash-saravanan-858113284/)
- **GitHub**: [@Prakash-M-S](https://github.com/Prakash-M-S)

*Happy to discuss machine learning fundamentals, implementation details, or career opportunities!*

---

**⭐ Star this repo if you found it helpful for learning ML fundamentals!**
