import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

filepath = r"C:\Users\hmnim\Desktop\OPINE Data Science\Python\ML1\Kaggle\NYC_Taxi\Latest\CleanedDataFiles\Combined_for_EDA.parquet"
cols = ['hour',"fare_amount", "trip_distance", 'PULocationID', 'DOLocationID']#"speed_mph", "day", "hour", "day_of_week"]
df = pd.read_parquet(filepath, columns=cols)
#df["tip_percentage"] = 100*df["tip_amount"]/df["fare_amount"]
df = df[df["fare_amount"] <= 1000]

'''
for i in range(24):
    df2 = df[df["day"] == i]
    plt.hist(df2["speed_mph"], log=True, histtype='step', label=f"Hour {i+1}", alpha=0.5)

plt.xlabel("Speed [mph]")
plt.ylabel("Frequency")
plt.legend()
plt.show()
'''

#df2 = df.groupby(["hour"])

## Pearson's median skewness coefficient: 3(mean - median) / S.D.
'''
skew_col = "fare_amount"
Skewness_Coefficient = 3*(df2[skew_col].mean() - df2[skew_col].median())/df2[skew_col].std()
print("Skewness_Coefficient:\n", Skewness_Coefficient)

plt.figure(figsize = (8,6))
plt.plot(range(24), Skewness_Coefficient, linewidth = 2, marker="p", color='purple', markersize=15)
plt.xticks(range(0,24,2), range(0,24,2))
plt.xlabel("Hour")
plt.ylabel("Skewness Coefficient")
plt.title(f"Hour-wise Skewness Coefficient on {skew_col} for January to November 2025")
#plt.savefig(f"skew_coefficient_{skew_col}.png", dpi=300)
plt.show()
'''

## Average Speed by Hour
'''
avg_speed = df2["speed_mph"].mean()
plt.plot(range(24), avg_speed, linewidth = 2, marker="o", color='skyblue', markersize=10)
plt.xlabel("Hour")
plt.ylabel("Average Speed")
plt.title("Avg Speed by Hour for January to November 2025")
#plt.savefig("Avg_speed_by_hour.png", dpi=300)
plt.show()
'''

'''
# 1. Aggregate your 32M rows into 168 rows (7 days * 24 hours)
# Assuming your df has columns: 'speed', 'hour', 'day_of_week'
df_agg = df.groupby(['day_of_week', 'hour'])['speed_mph'].mean().reset_index()
print(df_agg.head(10))

# a. Create a mapping dictionary
day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

# b. Map the integers to strings FIRST
df_agg['day_of_week'] = df_agg['day_of_week'].map(day_map)

# 2. Sort day_of_week logically (Monday to Sunday)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_agg['day_of_week'] = pd.Categorical(df_agg['day_of_week'], categories=day_order, ordered=True)
print(df_agg.head(10))

# 1. Calculate the mean speed per day (7 values)
day_averages = df_agg.groupby('day_of_week')['speed_mph'].mean()

# 2. Re-run your FacetGrid
g = sns.FacetGrid(df_agg, col="day_of_week", col_wrap=4, height=4, aspect=0.7)

# 3. Add the actual hourly data
g.map(sns.lineplot, "hour", "speed_mph", color="teal", linewidth=2)

# 4. Add the Global Hourly Avg (Dashed Gray)
overall_hourly_avg = df_agg.groupby('hour')['speed_mph'].mean()
for ax in g.axes.flat:
    ax.plot(overall_hourly_avg.index, overall_hourly_avg.values,
            color='gray', linestyle='--', alpha=0.4, label='Global Hourly Avg')

# 5. Add the Day-Specific Average (The Horizontal Line)
for ax, day in zip(g.axes.flat, day_order):
    day_mean = day_averages[day]
    ax.axhline(day_mean, color='red', linestyle='-', alpha=0.6, label='Day Avg')

    # Optional: Add a small text label for the average speed
    ax.text(0, day_mean + 1, f'{day_mean:.1f}', color='red', fontsize=9)

g.set_axis_labels("Hour", "Speed")
g.add_legend()
plt.tight_layout()
#plt.savefig("Cycle_Plot.png", dpi=1200)
plt.show()
'''

## Fare amount vs trip distance
#plt.hist(df["trip_distance"],log=True)
#plt.show()
'''
import matplotlib.colors as colors
plt.hist2d(df["trip_distance"], df["fare_amount"], bins=25, cmap='inferno', norm=colors.LogNorm())
plt.colorbar(label='Log10(Number of Trips)')
plt.xlabel('Distance (miles)')
plt.ylabel('Fare Amount ($)')
plt.xscale('log')
plt.yscale('log')
plt.title('Trip Density: Distance vs Fare')
plt.show()
'''
'''
from matplotlib.colors import LogNorm
plt.hexbin(df["trip_distance"], df["fare_amount"], gridsize=50, cmap='inferno', norm=LogNorm(), mincnt=10)
plt.title('Trip Density: Distance vs Fare')
plt.xlabel('Distance (miles)')
plt.ylabel('Fare Amount ($)')
plt.colorbar()
plt.show()
'''

## Binned median line
'''
bins = np.linspace(0, 30, 100)
df['distance_bin'] = pd.cut(df['trip_distance'], bins)

median_fare = df.groupby('distance_bin')['fare_amount'].median()
bin_centers = bins[:-1] + np.diff(bins)/2

plt.plot(bin_centers, median_fare)
plt.xlabel("Trip Distance")
plt.ylabel("Median Fare")
plt.title("Median Fare vs Distance")
#plt.savefig("Median_Fare_vs_Distance.png", dpi=300)
plt.show()
'''

most_trips_to = df["DOLocationID"].value_counts().head(20)
#print(most_trips_to)

most_trips_from = df[df["DOLocationID"]==most_trips_to.index[0]]['PULocationID'].value_counts().head(3)
#print(most_trips_from)
