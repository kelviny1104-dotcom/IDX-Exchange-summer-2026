# California Property Close Price Prediction

## Project Overview

This project develops a machine learning pipeline to predict the final sale price (`ClosePrice`) of California single-family residential properties using historical CRMLS transaction data.

The workflow covers exploratory data analysis, preprocessing, feature engineering, model comparison, XGBoost tuning, and expanded model evaluation. To reduce target leakage, listing-price fields such as `ListPrice` and `OriginalListPrice` are excluded from the modeling features.

---

## Dataset

### Source

The data consists of historical sold-property records from **CRMLS (California Regional Multiple Listing Service)** provided through the project data source. Monthly files follow the naming convention:

```text
CRMLSSoldYYYYMM.csv
```

The project uses data from **May 2025 through June 2026**.

### Modeling Scope

Only observations meeting all of the following conditions are retained:

- `StateOrProvince == "CA"`
- `PropertyType == "Residential"`
- `PropertySubType == "SingleFamilyResidence"`
- `ClosePrice` is present and greater than zero

`ClosePrice` is the prediction target.

The preprocessing notebook initially loads **306,330 rows and 78 columns**. After applying the modeling scope filters, **154,847 observations** remain.

---

## Exploratory Data Analysis

`Week2/01_exploration.ipynb` examines the distributions of major property and price variables, including:

- `ClosePrice`
- `LivingArea`
- `BedroomsTotal`
- `BathroomsTotalInteger`
- `LotSizeSquareFeet`

The notebook also reviews outliers and correlations among important numeric features.

---

## Data Preprocessing

Preprocessing is performed in `Week3/02_preprocessing.ipynb`.

### 1. Feature Selection

Columns that are identifiers, administrative fields, highly incomplete, redundant, overly granular, or likely to cause leakage are removed.

Important leakage exclusions include:

- `ListPrice`
- `OriginalListPrice`
- `DaysOnMarket`
- contract/listing-process date fields

Highly redundant fields are also reduced. For example, `LotSizeSquareFeet` is retained while `LotSizeAcres` and `LotSizeArea` are removed, and `Levels` is retained instead of `Stories`.

Columns with more than 60% missing values are dropped.

### 2. Invalid-Value Handling

Clearly invalid numeric values are converted to missing values rather than deleting the full observation. Examples include:

- invalid California latitude/longitude values
- non-positive living area
- very small or invalid lot-size values
- negative parking or garage values
- non-positive bedroom/bathroom counts
- inconsistent main-level bedroom counts
- unrealistic count-based values

### 3. Categorical Cleaning

Categorical features are standardized before encoding.

Examples include:

- `PostalCode` standardized to five-digit ZIP codes
- non-specific school-district values converted to missing
- flooring combinations converted into material indicator variables
- building-level combinations converted into structured level indicators
- boolean-like fields retained as categorical values so that missing values are not incorrectly treated as `False`

### 4. Time-Based Train/Test Split

The project uses a chronological split rather than a random split.

- **Training period:** May 2025 through May 2026
- **Test period:** June 2026
- **Training observations:** 141,823
- **Test observations:** 12,857

The most recent month is reserved as an unseen test set to better represent future prediction performance.

### 5. Target Outlier Handling

Target outlier rules are fitted and applied to the **training set only**.

The preprocessing removes:

- the extreme low end of `ClosePrice` using the 0.1% training quantile
- extreme high-price observations identified using abnormal price-per-living-area behavior

The June 2026 test set is not filtered.

### 6. Continuous Feature Capping

Upper caps are learned from the training set at the 99.9th percentile and then applied to both train and test data for:

- `LivingArea`
- `LotSizeSquareFeet`
- `AssociationFee`

### 7. Missing-Value Imputation, Scaling, and Encoding

Numeric variables use:

- `StandardScaler`
- `KNNImputer`
  - `n_neighbors = 5`
  - distance weighting
  - missing indicators

Categorical variables use:

- constant imputation with `"Unknown"`
- `OneHotEncoder`
- infrequent-category handling with a minimum frequency of 50

The fitted preprocessing objects are saved in:

```text
Week3/preprocessing_artifacts.joblib
```

The preprocessing step also produces:

```text
Week3/model_data_knn_before_encoding.csv
Week3/model_data_knn_after_encoding.csv.gz
```

---

## Feature Engineering

The updated feature set adds information that is not explicitly represented in the original variables.

### Property-Level Features

- **PropertyAge**  
  Calculated from the sale year and `YearBuilt`.

- **BedBathRatio**  
  Calculated as bedrooms divided by bathrooms, with zero-bathroom cases handled as missing.

### Geographic Feature

A more detailed geographic feature is created by spatially joining reliable property coordinates to California **Unified School District** boundary polygons.

GeoPandas is used for the spatial join. Properties without a reliable district match are assigned to an `Unknown` category.

The updated features improved the Linear Regression model, while tree-based models changed only slightly.

---

## Models Tested

The project compares the following regression models:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Baseline XGBoost Regressor
5. Tuned XGBoost Regressor

Time-based cross-validation is used for tree-based model selection and XGBoost tuning.

### Model Comparison

| Model | Test R² |
|---|---:|
| Linear Regression — original features | 0.4714 |
| Decision Tree — original features | 0.4777 |
| Random Forest — original features | 0.5168 |
| Linear Regression — engineered features | 0.4854 |
| Decision Tree — engineered features | 0.4768 |
| Random Forest — engineered features | 0.5154 |
| Baseline XGBoost — engineered features | 0.5192 |
| **Tuned XGBoost — engineered features** | **0.5454** |

---

## Best Model

The strongest model in the reproducible model-comparison workflow is the **tuned XGBoost regressor**.

Best hyperparameters:

```text
learning_rate = 0.15
max_depth = 6
n_estimators = 400
```

Performance:

- **Cross-validation R²:** 0.7185
- **Training R²:** 0.9476
- **Test R²:** 0.5454

The tuned model improves test R² by approximately **0.0262** over the baseline XGBoost model, although the train/test gap indicates that overfitting remains an area for improvement.

---

## Expanded Evaluation

The final evaluation summary in `metrics_summary.csv` evaluates the June 2026 test set with additional error metrics.

### Overall June 2026 Evaluation

| Metric | Result |
|---|---:|
| Properties | 12,857 |
| R² | 0.8105 |
| MAE | $224,476 |
| RMSE | $668,791 |
| MAPE | 24.08% |
| MdAPE | 11.30% |

A sensitivity analysis excluding the lowest 1% of actual close prices reduces MAPE to approximately **15.53%**, showing that percentage-based error is strongly affected by unusually low-price transactions.

Across most middle and upper price bands, median absolute percentage error remains substantially lower than the overall MAPE.

> Note: the model-selection results and the expanded evaluation artifact are reported separately because they were produced at different stages of the project workflow.

## Key Takeaways

- A chronological June 2026 holdout set provides a more realistic future-data evaluation than a random split.
- Random Forest improves substantially over the original Linear Regression baseline.
- Engineered property-age, bed/bath, and school-district features benefit the linear model but add relatively little to the tree ensembles.
- Tuned XGBoost provides the strongest performance in the model-comparison stage.
- Error varies significantly by price segment, and extremely low sale prices have a disproportionate effect on percentage-based metrics.
