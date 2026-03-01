import pandas as pd
import glob
import os

class YellowTaxiCompiler:

    def __init__(self, input_folder, output_file):
        self.input_folder = input_folder
        self.output_file = output_file

    def compile(self):
        files = sorted(glob.glob(os.path.join(self.input_folder, "*.parquet")))
        print(f"Found {len(files)} files.")

        if not files:
            print("No files found.")
            return

        df = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)

        print("Total rows after combining:", len(df))
        df.to_parquet(self.output_file, index=False)
        print("Combined file saved successfully.")