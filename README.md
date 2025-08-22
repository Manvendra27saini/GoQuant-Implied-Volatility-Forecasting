# Implied Volatility Forecasting Pipeline

###  Overview

This project presents a robust and comprehensive machine learning pipeline for forecasting **10-second-ahead implied volatility (IV)** for the ETH cryptocurrency, referred to as "QCoin." The solution is designed for high-frequency time-series data and is optimized for performance and reproducibility within a Kaggle Notebook environment.

The core of the solution is a powerful **ensemble model** that leverages advanced feature engineering and a rigorous validation strategy to achieve high predictive accuracy. The primary performance metric is the **Pearson Correlation Score**.

---

### Code Documentation & Pipeline Breakdown

#### 1. Data Loading (`try_load_data` function)

This section is responsible for safely loading the required datasets from the Kaggle environment.

- `try_load_data()`: A helper function that attempts to load `train/ETH.csv` and `test/ETH.csv`. It includes multiple file paths to handle different directory structures, ensuring the notebook runs without modification.
- **Error Handling**: The code includes explicit checks to ensure data is loaded correctly and that the `label` column exists, preventing pipeline failures.
- **Initial Visualization**: A plot of the first 2000 `mid_price` data points is generated to provide a quick visual check of the data's integrity and characteristics.

#### 2. Feature Engineering (`create_robust_features` function)

This is the most critical part of the pipeline, where raw data is transformed into a wide array of meaningful predictive signals based on market microstructure theory. The pipeline generates over **150 features**, including:

- **Price and Return Metrics**: Log returns and price momentum features to capture price dynamics.
- **Spread & Order Book Features**: Metrics for bid-ask spreads at multiple levels, indicating market liquidity.
- **Volume and VWAP (Volume-Weighted Average Price)**: Features that measure market activity, total volume, and directional pressure through order imbalance.
- **Rolling Window Statistics**: The most important features, including **realized volatility** over multiple time windows (e.g., 3s, 5s, 10s, 20s, 50s, 100s), which are strong predictors of future volatility.
- **Technical Indicators**: Features derived from classic indicators like the **Relative Strength Index (RSI)** and **Bollinger Bands**.
- **Interaction & Lagged Features**: New features created by multiplying existing ones to capture non-linear relationships, as well as lagged values of key variables to incorporate temporal dependencies.

#### 3. Target Transformation

This section prepares the target variable (`label`) for training. The pipeline evaluates several transformations to find the best representation for the model.

- `create_target_transforms()`: Generates different versions of the target variable, including `original`, `log`, `sqrt`, `quantile_normal`, `smooth_3`, and `smooth_5`.
- **Selection Process**: A base LGBM model is trained and evaluated on each transformed target. The transformation that achieves the highest Pearson correlation is chosen for the final ensemble.

#### 4. Time-Series Validation (`TimeSeriesSplit`)

- **Preventing Data Leakage**: To simulate a real-world forecasting scenario, `TimeSeriesSplit` is used. This validation method ensures that training data always precedes validation data, preventing the model from "looking ahead" and providing a reliable estimate of its out-of-sample performance.

#### 5. Ensemble Modeling (`EnhancedEnsemble` class)

The pipeline's core forecasting engine uses an ensemble of diverse models to produce a more robust and accurate prediction.

- `create_models()`: Initializes three `LGBMRegressor` models with different hyperparameters and, if available, a `CatBoostRegressor`.
- `fit()`: Trains each model and calculates weights based on their individual validation correlation scores. Models with higher scores contribute more to the final prediction.
- `predict()`: Generates predictions from each individual model on the test set and combines them using the calculated weights.

#### 6. Prediction, Submission, and Summary

- **Prediction**: The ensemble generates final predictions for the test data.
- **Clipping**: Predictions are clipped to the 1st and 99th percentiles of the training target. This is a critical post-processing step that prevents extreme or unrealistic predictions.
- **Submission**: The final predictions are saved to `submission.csv` in the required format.
- **Visualization**: The pipeline generates plots to visually summarize performance, including time-series plots, distribution comparisons, and a scatter plot of true vs. predicted values.

---

###  Final Results & Performance Analysis

The final run of the IV forecasting pipeline delivered exceptional results, confirming the effectiveness of the chosen methodology. The model's ability to capture complex market dynamics is demonstrated by the strong Pearson Correlation scores.

### **Key Performance Metrics**

- **Final Ensemble Validation Correlation**: **0.7004** 
  - This score indicates a very strong linear relationship between the model's predictions and the true IV values on the validation set. A value over 0.7 is a highly impressive result in financial forecasting.
- **Best Single Model Validation Correlation**: **0.6765**
  - This score, achieved by an individual LGBM model, serves as a strong baseline. The lift to the ensemble's 0.7004 highlights the value of combining diverse models to reduce variance and improve performance.
- **Selected Target Transformation**: **`sqrt`**
  - The data-driven approach correctly identified the square root transformation as the most effective for this dataset, leading to a significant performance boost over the original target variable (0.6157).
- **Pipeline Runtime**: **32.49 minutes**
  - The total runtime is a fair trade-off for the comprehensive feature engineering and the training of multiple robust models.

### **Requirements**

The pipeline can be run directly within a Kaggle Notebook. The necessary libraries are automatically installed if not already present.

- `pandas`
- `numpy`
- `matplotlib`
- `lightgbm`
- `catboost` (optional, but highly recommended)
- `scikit-learn`
- `scipy`

### **How to Run**

1.  Open this notebook on Kaggle.
2.  Ensure the necessary datasets are linked via the competition page.
3.  Run all cells. The script will automatically perform data loading, feature engineering, model training, and generate the `submission.csv` file.

### **Author**

This solution was developed by Manvendra Saini. If you have any questions, please contact manvendrasaini2005@gmail.com.
