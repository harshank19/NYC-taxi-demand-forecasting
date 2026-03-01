import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

filepath = r"C:\Users\hmnim\Desktop\OPINE Data Science\Python\ML1\Kaggle\NYC_Taxi\Latest\CleanedDataFiles\combined_2025.parquet"
cols = ["fare_amount", 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID']
df = pd.read_parquet(filepath, columns=cols)
#print(len(df))
df = df[df["fare_amount"] <= 1000]
#print(len(df))

df['pickup_hour'] = df['tpep_pickup_datetime'].dt.floor('h')

df = df.groupby(['PULocationID', 'pickup_hour']).size()
df = df.reset_index(name='trip_count')
df['hour'] = df['pickup_hour'].dt.hour
df['day_of_week'] = df['pickup_hour'].dt.dayofweek
df['weekend_flag'] = np.where(df['day_of_week'] >= 5, 1, 0)
df2 = df.sort_values(['PULocationID', 'pickup_hour'])

df2_feature_set = df2.copy()

df2_feature_set['hourly_lag'] = df2.groupby('PULocationID')['trip_count'].shift(1)
df2_feature_set['24_hourly_lag'] = df2.groupby('PULocationID')['trip_count'].shift(24)

df2_feature_set = df2_feature_set.dropna()

df2_feature_set['3_rolling_mean_lag'] = df2_feature_set['hourly_lag'].rolling(3).mean()
df2_feature_set['6_rolling_mean_lag'] = df2_feature_set['hourly_lag'].rolling(6).mean()

df2_feature_set = df2_feature_set.dropna()

#print(df2_feature_set["pickup_hour"].min())
#print(df2_feature_set["pickup_hour"].max())

df_train = df2_feature_set[df2_feature_set["pickup_hour"] < '2025-10-01']
df_test = df2_feature_set[df2_feature_set["pickup_hour"] >= '2025-10-01']

#print(len(df_train), len(df_test))

locations = df2_feature_set["PULocationID"].unique()
#print(locations)

location_wise_train_sets = []
location_wise_test_sets = []
models = []
total_trips = {}

for i in locations:
    location_df = df2_feature_set[df2_feature_set["PULocationID"] == i]
    trips = location_df["trip_count"].sum()
    if trips > 500000:
        total_trips[i] = trips
        location_wise_train_sets.append(df_train[df_train["PULocationID"] == i])
        location_wise_test_sets.append(df_test[df_test["PULocationID"] == i])
    else:
        continue
print(sum(total_trips.values()))
'''
zone_ids = [i.item() for i in total_trips.keys()][:25]
print(zone_ids)

from sklearn.ensemble import RandomForestRegressor

X_train = []
y_train = []
X_test = []
y_test = []
for i in range(len(location_wise_train_sets)):
    X_train.append(location_wise_train_sets[i].drop(["PULocationID", "trip_count", "pickup_hour"], axis=1))
    y_train.append(location_wise_train_sets[i].trip_count)
    X_test.append(location_wise_test_sets[i].drop(["PULocationID", "trip_count", "pickup_hour"], axis=1))
    y_test.append(location_wise_test_sets[i].trip_count)
    models.append(RandomForestRegressor(random_state=1761))

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import math

results = []
feature_importance_list = []
all_predictions = []

for i in range(len(models)):
    model = models[i]
    model.fit(X_train[i], y_train[i])
    joblib.dump(model,f"rf_zone_{zone_ids[i]}.pkl")
    importances = model.feature_importances_
    feature_names = model.feature_names_in_
    y_test_predicted = model.predict(X_test[i])
    y_train_predicted = model.predict(X_train[i])
    mae_train = mean_absolute_error(y_train[i], y_train_predicted)
    mae_test = mean_absolute_error(y_test[i], y_test_predicted)
    rmse_train = math.sqrt(mean_squared_error(y_train[i], y_train_predicted))
    rmse_test = math.sqrt(mean_squared_error(y_test[i], y_test_predicted))
    r2_train = r2_score(y_train[i], y_train_predicted)
    r2_test = r2_score(y_test[i], y_test_predicted)
    zone_result = {
        "zone_id": zone_ids[i],
        "train_rows": len(X_train[i]),
        "test_rows": len(X_test[i]),
        "mean_y_test": y_test[i].mean(),
        "std_dev_y_test" : y_test[i].std(),
        "MAE_train": mae_train,
        "MAE_test": mae_test,
        "RMSE_train": rmse_train,
        "RMSE_test": rmse_test,
        "R2_train": r2_train,
        "R2_test": r2_test,
        "Relative_MAE": mae_test / y_test[i].mean()
    }
    results.append(zone_result)
    fi_dict = dict(zip(feature_names, importances))
    fi_dict["zone_id"] = zone_ids[i]
    feature_importance_list.append(fi_dict)
    temp_df = pd.DataFrame({
        "zone_id": zone_ids[i],
        "pickup_hour": location_wise_test_sets[i]["pickup_hour"].values,
        "actual": y_test[i].values,
        "predicted": y_test_predicted
    })
    all_predictions.append(temp_df)

results_df = pd.DataFrame(results)
results_df.to_csv("zone_model_summary.csv", index=False)

fi_df = pd.DataFrame(feature_importance_list)
fi_df.to_csv("feature_importance_by_zone.csv", index=False)

predictions_df = pd.concat(all_predictions, ignore_index=True)
predictions_df.to_csv("all_zone_test_predictions.csv", index=False)
'''