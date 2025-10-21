# Linear Regression: Bike Sharing Prediction 🚴‍♂️

## Overview
Implementation of **Linear Regression from scratch** following Andrew Ng's Machine Learning Course concepts, applied to predict daily bike rental counts. This project demonstrates core ML fundamentals including gradient descent, cost functions, and feature scaling.

## Dataset
- **Source**: Daily bike sharing dataset with weather and seasonal features
- **Features**: Season, year, month, holiday, working day, weather situation, temperature, humidity, wind speed
- **Target**: Daily bike rental count
- **Size**: 731 days of data

## Key Implementations

### 🔧 From Scratch Implementation
- **Cost Function**: Mean Squared Error (MSE)
- **Gradient Descent**: Custom implementation with learning rate α = 0.01
- **Feature Scaling**: Z-score normalization for both features and target
- **Train/Test Split**: 80/20 random split with seed=42

### 📊 Scikit-learn Comparison
Direct performance comparison between custom implementation and sklearn's LinearRegression.

## Results

| Metric | From Scratch | Scikit-learn |
|--------|-------------|--------------|
| MAE    | 651.99 | 651.96 |
| RMSE   | 859.36 | 857.62 |

*Results show excellent agreement between custom implementation and sklearn, validating the correctness of the from-scratch gradient descent algorithm.*

## Visualizations
- **Learning Curve**: Cost function decrease over iterations
- **Predicted vs Actual**: Scatter plot showing model performance
- **Regression Line**: Model fit visualization

## Key Learnings from Andrew Ng's Course
1. **Gradient Descent Mechanics**: Understanding how parameters update iteratively
2. **Feature Scaling Importance**: Normalized features improve convergence
3. **Cost Function Interpretation**: MSE provides intuitive loss measurement
4. **Overfitting Prevention**: Proper train/test split validation

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
```
linear_regression_bikes/
├── README.md
├── data/
│   └── day.csv              # Bike sharing dataset
├── notebooks/
│   └── scratch.ipynb        # Main implementation notebook
└── results/
    ├── cost_curve.png       # Learning curve visualization
    ├── predicted_vs_actual.png  # Model performance plot
    └── regression_line.png   # Regression fit visualization
```

## Andrew Ng Course Concepts Demonstrated
- ✅ **Linear Regression Theory**: Mathematical foundation
- ✅ **Gradient Descent**: Parameter optimization algorithm
- ✅ **Feature Normalization**: Preprocessing for better convergence
- ✅ **Cost Function**: MSE loss function implementation
- ✅ **Model Evaluation**: Performance metrics and visualization

## Advanced Features Implemented

- ✅ **Ridge Regularization**: L2 penalty to prevent overfitting
- ✅ **Cross-Validation**: 5-fold CV for robust model evaluation  
- ✅ **Overfitting Analysis**: Comprehensive diagnostics with multiple validation methods
- ✅ **Learning Curve Analysis**: Cost function convergence monitoring

## Next Steps

- [ ] Feature engineering and polynomial features
- [ ] Learning rate optimization (adaptive methods)
- [ ] Lasso (L1) regularization implementation

---
*This project demonstrates practical application of Andrew Ng's Machine Learning Course fundamentals. The from-scratch implementation provides deep understanding of underlying algorithms before using high-level libraries.*
