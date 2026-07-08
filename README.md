# Melbourne Housing Price Prediction

An end-to-end data analytics project: cleaning, exploratory analysis, feature engineering, and a regression model that predicts house prices in Melbourne, Australia.


## Dataset

**Melbourne Housing Snapshot**

- Source (Kaggle): https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
- Original full dataset: https://www.kaggle.com/anthonypino/melbourne-housing-market
- Snapshot created: September 2017
- Rows without a `Price` value were removed from the snapshot
- Target variable: `Price`

### Columns

| Column | Description |
|---|---|
| `Address` | Property address |
| `Type` | `h` – house/cottage/villa/semi/terrace, `u` – unit/duplex, `t` – townhouse, `dev site` – development site, `br` – bedroom(s), `o res` – other residential |
| `Suburb` | Suburb name |
| `Method` | Sale method: `S` – sold, `SP` – sold prior, `PI` – passed in, `PN` – sold prior not disclosed, `SN` – sold not disclosed, `NB` – no bid, `VB` – vendor bid, `W` – withdrawn prior to auction, `SA` – sold after auction, `SS` – sold after auction, price not disclosed |
| `Rooms` | Number of rooms |
| `Price` | Sale price, in dollars |
| `SellerG` | Real estate agent |
| `Date` | Date of sale |
| `Distance` | Distance from CBD (Central Business District) |
| `Regionname` | General region (e.g. Northern Metropolitan, Southern Metropolitan) |
| `Propertycount` | Number of properties in the suburb |
| `Bedroom2` | Number of bedrooms (scraped from a different source than `Rooms`) |
| `Bathroom` | Number of bathrooms |
| `Car` | Number of car spots |
| `Landsize` | Land size |
| `BuildingArea` | Building size |
| `YearBuilt` | Year the property was built |
| `CouncilArea` | Governing council for the area |
| `Lattitude` / `Longtitude` | Geographic coordinates |

## What was done

1. **Loading** — read `melb_data.csv` into a DataFrame; verified shape and column types.
2. **Initial overview** — checked missing values, data types, and descriptive statistics; found data quality issues (invalid `YearBuilt` values, `Date` stored as text).
3. **Cleaning**
   - Removed technical index columns, if present.
   - Converted `Date` from text to a proper date type (day-first format).
   - Treated `Landsize` / `BuildingArea` values of `0` as missing (a real property can't have zero area).
   - Treated `YearBuilt` values below 1800 as missing (data entry errors, e.g. `1196`).
   - Checked and removed duplicate rows.
4. **Exploratory data analysis (EDA)**
   - Distribution of `Price` (right-skewed; log10 transform is closer to symmetric).
   - Price by dwelling type (`Type`).
   - Correlation of numeric features with `Price`.
   - Identified near-duplicate features (`Rooms` and `Bedroom2`, correlation 0.94).
5. **Feature engineering**
   - `sale_year`, `sale_month` — extracted from `Date`.
   - `building_age` — age of the property at time of sale (`sale_year - YearBuilt`), with negative values treated as missing.
   - `has_buildingarea`, `has_yearbuilt` — missingness flags.
   - `suburb_freq` — frequency encoding for the high-cardinality `Suburb` column.
   - Dropped `Bedroom2` (redundant with `Rooms`), plus raw text columns not used for modeling (`Address`, `SellerG`, `Suburb`, `Date`).
6. **Modeling pipeline**
   - Numeric features: median imputation.
   - Categorical features (`Type`, `Method`, `Regionname`, `CouncilArea`): most-frequent imputation + one-hot encoding.
   - All preprocessing steps combined into a single `scikit-learn` `Pipeline` / `ColumnTransformer` to prevent data leakage between train and test sets.
   - 80/20 train/test split (`random_state=42`).
7. **Models trained and evaluated on the test set**
   - Baseline (`DummyRegressor`, predicts the mean price)
   - Linear Regression
   - Random Forest Regressor (200 trees)

## Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline (mean) | 461,258 | 630,259 | 0.00 |
| Linear Regression | 256,130 | 372,664 | 0.65 |
| **Random Forest** | **162,919** | **269,667** | **0.82** |

*(MAE / RMSE in dollars.)*

**Top price drivers** (Random Forest feature importance): region (`Southern Metropolitan`), `Rooms`, `Distance` to CBD, dwelling type `unit`, `Landsize`.

## Conclusions

- Random Forest clearly outperforms both the baseline and Linear Regression, explaining about 82% of price variance versus 65% for a plain linear model.
- Location matters most: region and distance to the CBD are among the strongest predictors, alongside the number of rooms and dwelling type.
- Houses (`h`) sell for a substantially higher median price than townhouses (`t`) or units (`u`), consistent with the correlation and feature-importance results.

## Ideas for improvement

- Hyperparameter tuning for Random Forest (`GridSearchCV` / `RandomizedSearchCV`).
- Try gradient boosting models (XGBoost, LightGBM, CatBoost).
- Engineer geospatial features from `Lattitude` / `Longtitude` (e.g. distance to CBD center, location clustering).
- Handle `Suburb` and `CouncilArea` with target/mean encoding instead of, or in addition to, frequency encoding.

