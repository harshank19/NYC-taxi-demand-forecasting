import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"Results\zone_model_summary.csv")
print(df.columns)

#df = df.drop(columns =["MAE_train", "RMSE_train", "R2_train"])
#df["Coefficient_of_variation"] = 100*(df["std_dev_y_test"].values/df["mean_y_test"].values)

#print(df.columns)
#df.to_csv(r"Results\Master_summary_table.csv", index=False)
#print(df["R2_test"].min(), df["R2_test"].max(), df["R2_test"].mean())

df2 = pd.read_csv(r"Results\all_zone_test_predictions.csv")
zones = df2["zone_id"].unique()
total_trips = []

for i in zones:
    df3 = df2.loc[df2["zone_id"] == i, ["actual"]]
    total_trips.append(np.sum(df3["actual"].values).item())


print(sum(total_trips))
'''
df3 = df2.loc[df2["zone_id"] == 48, ["actual"]]
total = np.sum(df3["actual"].values)
print(total)
'''