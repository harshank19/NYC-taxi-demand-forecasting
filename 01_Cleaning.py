from Cleaning.NYC_Cleaner import YellowTaxiCleaner
from Cleaning.NYC_YellowTaxi_Compiler import YellowTaxiCompiler
import glob
import os

input_folder = r"C:\Users\hmnim\Desktop\OPINE Data Science\Python\ML1\Kaggle\NYC_Taxi\Latest\RawDataFiles"
output_folder = r"C:\Users\hmnim\Desktop\OPINE Data Science\Python\ML1\Kaggle\NYC_Taxi\Latest\CleanedDataFiles"

files = sorted(glob.glob(os.path.join(input_folder, "*.parquet")))

cleaner = YellowTaxiCleaner()

global_stats = {}
global_total_rows = 0

for file in files:
    filename = os.path.basename(file)
    output_path = os.path.join(output_folder, "cleaned_" + filename)

    print(f"Processing {filename}...")
    report = cleaner.run(file, output_path)

    for key, value in report.items():
        if key not in global_stats:
            global_stats[key] = 0
        global_stats[key] += value

    global_total_rows += report["total_rows"]

print("\n====== GLOBAL ANOMALY REPORT ======")

for key, value in global_stats.items():
    if key != "total_rows":
        print(f"{key}: {value} ({100 * value / global_total_rows:.2f}%)")

print("Total rows across all files:", global_total_rows)

import pandas as pd

pd.DataFrame(global_stats.items(), columns=["Anomaly", "Count"]) \
  .to_csv("anomaly_summary.csv", index=False)

# Compile all cleaned files
compiler = YellowTaxiCompiler(
    input_folder=output_folder,
    output_file=os.path.join(output_folder, "combined_2025.parquet")
)

compiler.compile()

