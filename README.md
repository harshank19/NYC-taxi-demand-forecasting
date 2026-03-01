# NYC Taxi Zone-Level Demand Forecasting

Scalable spatiotemporal forecasting framework for hourly taxi demand using 31.86M NYC Yellow Taxi trips.

Full Report: [NYC_Taxi_Report.pdf](https://drive.google.com/file/d/1iw8-M8qsIAONGD-YZX7EYHDpqMJ35BII/view?usp=sharing)

## Overview

This project builds a high-resolution hourly demand forecasting system for NYC taxi zones using structured temporal feature engineering and ensemble regression.

Trips were aggregated at the **zone-hour level** and modeled independently across 25 high-volume zones. A chronological train-test split was used to simulate real-world forecasting conditions.

Random Forest models achieved:

* **Average Test R²: 0.92**
* **Best Zone R²: 0.9676**
* **Worst Zone R²: 0.8031**
* **Average Relative MAE: 12.6%**

The results demonstrate that structured temporal signals alone can produce strong generalization performance across heterogeneous urban demand profiles .

---

## Problem Statement

Accurate short-term demand forecasting is critical for:

* Fleet allocation
* Surge pricing
* Driver positioning
* Real-time dispatch systems

The objective is to forecast **hourly pickup demand per taxi zone** using only historical pickup timestamps and location identifiers .

---

## Dataset

* NYC Yellow Taxi Trip Records
* Raw trips: 44.4M
* Cleaned trips used: 31.86M
* Time span: Jan–Nov 2025
* Aggregated observations: ~800,000 zone-hour rows across 25 high-volume zones 

---

## Data Engineering Pipeline

An object-oriented preprocessing framework was implemented to:

* Clean and validate raw taxi records
* Detect anomalies
* Aggregate multi-file datasets
* Generate diagnostic reports

Zone-level aggregation was performed by:

* Pickup Location ID
* Hourly timestamp (floored to hour)

This preserves temporal causality while reducing dimensionality .

---

## Feature Engineering

All features were constructed post-aggregation to prevent leakage.

### Short-Term Momentum

* 1-hour lag demand

### Daily Seasonality

* 24-hour lag demand
* Hour of day
* Day of week
* Weekend indicator

### Trend Smoothing

* 3-hour rolling mean
* 6-hour rolling mean

These features capture intra-day cycles, daily seasonality, and short-term demand persistence .

---

## Modeling Strategy

* Independent Random Forest Regressors per zone
* Chronological split:

  * Train: Jan–Sept
  * Test: Oct–Nov
* Target: Hourly trip count

Zone-wise modeling was chosen to:

* Capture heterogeneous demand behavior
* Avoid artificial ordinal encoding of zone IDs
* Enable parallel scalability .

---

## Evaluation Metrics

Models were evaluated on out-of-sample data using:

* MAE
* RMSE
* R²
* Relative MAE (MAE / mean demand)

This ensures both absolute and scale-normalized error interpretation .

---

## Performance Summary (Test Set)

| Metric       | Best Zone | Median Zone | Worst Zone | Average (25 Zones) |
| ------------ | --------- | ----------- | ---------- | ------------------ |
| R²           | 0.9676    | 0.9287      | 0.8031     | 0.9223             |
| MAE          | 18.37     | 11.31       | 31.94      | 15.70              |
| RMSE         | 28.17     | 16.83       | 47.01      | 23.54              |
| Relative MAE | 9.2%      | 11.5%       | 19.4%      | 12.6%              |

The framework generalizes well across diverse demand densities .

---

## Residual Diagnostics

Residual analysis revealed:

* Increased variance during peak hours
* Stable, unbiased predictions in high-volume zones
* Systematic overprediction in high-demand regimes for certain sparse zones

This suggests that hierarchical or spatially-aware modeling may further improve robustness .

---

## Feature Importance Insights

* Hourly lag dominates predictive performance
* Hour-of-day encodes strong cyclical structure
* 24-hour lag captures daily seasonality

Temporal momentum and seasonality are primary drivers of demand predictability .

---

## Computational Considerations

* 31.86M cleaned rows pre-aggregation
* 23.5M trips covered by selected 25 zones (74% of total demand)
* Model artifacts serialized for reproducibility
* Scalable design supports parallel zone training .

---

## Limitations

* No exogenous variables (weather, events)
* Independent zone modeling ignores spatial correlations
* Performance degradation in sparse zones .

---

## Future Work

* Weather and event integration
* Spatial graph-based models
* Gradient boosting / probabilistic forecasting
* Recursive real-time deployment pipeline .
