########################################################################################################################
import pandas as pd

class YellowTaxiCleaner:
    def __init__(self, max_duration = 10000, min_speed = 2, max_speed = 100):
        self.df = None
        self.max_duration_seconds = max_duration
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.anomaly_stats = {}
        self.total_rows_before = 0

    def load_data(self, filepath):
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")
        self.df = pd.read_parquet(filepath)
        self.initial_rows = len(self.df)

    def anomaly_report(self):
        df = self.df.copy()
        total_rows = len(df)

        duration = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
        speed = df["trip_distance"] / (duration / 3600)

        conditions = {
            "missing_values": df[[
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "trip_distance",
                "fare_amount",
                "passenger_count"
            ]].isna().any(axis=1),

            "zero_distance": df["trip_distance"] <= 0,
            "zero_fare": df["fare_amount"] <= 0,
            "zero_passenger": df["passenger_count"] <= 0,
            "invalid_time_order": df["tpep_dropoff_datetime"] <= df["tpep_pickup_datetime"],
            "too_short_trip": duration <= 60,
            "too_long_trip": duration >= self.max_duration_seconds,
            "too_slow": speed <= self.min_speed,
            "too_fast": speed >= self.max_speed,
        }

        report = {}
        union_mask = False

        for name, mask in conditions.items():
            count = mask.sum()
            report[name] = count
            union_mask |= mask

        report["total_flagged_rows"] = union_mask.sum()
        report["total_rows"] = total_rows

        self.anomaly_stats = report

        return report

    def basic_cleaning(self):
        df = self.df

        df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "fare_amount", "passenger_count"])

        df = df[df["trip_distance"] > 0]
        df = df[df["fare_amount"] > 0]
        df = df[df["passenger_count"] > 0]
        df = df[df["tpep_dropoff_datetime"] > df["tpep_pickup_datetime"]]

        self.df = df.copy()


    def feature_engineering(self):
        df = self.df

        df["trip_duration_seconds"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()

        df = df[(df["trip_duration_seconds"] > 60) &
                (df["trip_duration_seconds"] < self.max_duration_seconds)]

        df["speed_mph"] = df["trip_distance"] / (df["trip_duration_seconds"]/3600)
        df = df[(df["speed_mph"] > self.min_speed) &
                (df["speed_mph"] < self.max_speed)]

        df["year"] = df["tpep_pickup_datetime"].dt.year
        df["month"] = df["tpep_pickup_datetime"].dt.month
        df["day"] = df["tpep_pickup_datetime"].dt.day
        df["hour"] = df["tpep_pickup_datetime"].dt.hour
        df["day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        self.df = df

    def save_data(self, outputfilepath):
        self.df.to_parquet(outputfilepath, index=False)

    def summary(self):
        final_rows = len(self.df)
        print(f"Rows before: {self.initial_rows}")
        print(f"Rows after: {final_rows}")
        print(F"Removed : {self.initial_rows - final_rows}")

    def run(self, inputfilepath, outputfilepath=None):
        self.load_data(inputfilepath)

        report = self.anomaly_report()

        self.basic_cleaning()
        self.feature_engineering()
        self.summary()

        if outputfilepath:
            self.save_data(outputfilepath)

        return report

########################################################################################################################