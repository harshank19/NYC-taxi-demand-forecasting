import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

worst_R2_zone = 138
median_R2_zone = 141
best_R2_zone = 236

df = pd.read_csv(r"Results\all_zone_test_predictions.csv")

worst_R2_df = df[df["zone_id"] == worst_R2_zone]
median_R2_df = df[df["zone_id"] == median_R2_zone]
best_R2_df = df[df["zone_id"] == best_R2_zone]

worst_R2_df = worst_R2_df.loc[worst_R2_df['pickup_hour'] < '2025-10-15']
median_R2_df = median_R2_df.loc[median_R2_df['pickup_hour'] < '2025-10-15']
best_R2_df = best_R2_df.loc[best_R2_df['pickup_hour'] < '2025-10-15']

## Worst Performing Zone: Zone ID 138
positions = list(range(3, len(worst_R2_df), 24))
labels = pd.to_datetime(worst_R2_df["pickup_hour"])
labels = labels.dt.strftime('%B %d')
labels = labels.iloc[::24]

plt.plot(worst_R2_df['pickup_hour'], worst_R2_df['actual'], label="Actual Demand", linewidth=2, color='#0072B2')
plt.plot(worst_R2_df['pickup_hour'], worst_R2_df['predicted'], label="Predicted Demand", linewidth=2, color='#D55E00')
plt.xticks(positions,labels,rotation=45)
plt.title(f'Worst performing zone: {worst_R2_zone}')
plt.legend()
plt.show()


## Best Performing Zone: Zone ID 236
positions = list(range(11, 24*14, 24))
labels = pd.to_datetime(best_R2_df["pickup_hour"])
labels = labels.dt.strftime('%B %d')
labels = labels.iloc[::24]

plt.plot(best_R2_df['pickup_hour'], best_R2_df['actual'], label="Actual Demand", linewidth=2, color='#0072B2')
plt.plot(best_R2_df['pickup_hour'], best_R2_df['predicted'], label="Predicted Demand", linewidth=2, color='#009E73')
plt.xticks(positions,labels,rotation=45)
plt.title(f'Best performing zone: {best_R2_zone}')
plt.legend()
plt.show()


## Median Performing Zone: Zone ID 141
positions = list(range(11, 24*14, 24))
labels = pd.to_datetime(median_R2_df["pickup_hour"])
labels = labels.dt.strftime('%B %d')
labels = labels.iloc[::24]

plt.plot(median_R2_df['pickup_hour'], median_R2_df['actual'], label="Actual Demand", linewidth=2, color='#0072B2')
plt.plot(median_R2_df['pickup_hour'], median_R2_df['predicted'], label="Predicted Demand", linewidth=2, color='#E69F00')
plt.xticks(positions,labels,rotation=45)
plt.title(f'Median performing zone: {median_R2_zone}')
plt.legend()
plt.show()


