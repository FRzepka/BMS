import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

base_path = "/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes"

def scale_voltage(voltage):
    """Scales voltage to the range of 0 to 1 using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(voltage.reshape(-1, 1)).flatten()

def scale_current(current):
    """Scales current to the range of -1 to 1 using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(current.reshape(-1, 1)).flatten()

def scale_temperature(temperature):
    """Scales temperature to the range of 0 to 1 using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(temperature.reshape(-1, 1)).flatten()

print("Collecting data from subdirectories...")
subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

dfs_by_path = {}
for d in tqdm(subdirs, desc="Loading df.parquet"):
    parquet_path = os.path.join(base_path, d, "df.parquet")
    if not os.path.isfile(parquet_path):
        continue
    print(f"Reading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    dfs_by_path[d] = df

print("Applying individual scalers and saving df_scaled.parquet...")
for d in tqdm(dfs_by_path, desc="Scaling"):
    df = dfs_by_path[d].copy()

    # Apply individual scaling to each column
    df["Voltage[V]"]     = scale_voltage(df["Voltage[V]"].values)
    df["Current[A]"]     = scale_current(df["Current[A]"].values)
    df["Temperature[°C]"] = scale_temperature(df["Temperature[°C]"].values)
    
    df.rename(
        columns={
            "Voltage[V]": "Scaled_Voltage[V]",
            "Current[A]": "Scaled_Current[A]",
            "Temperature[°C]": "Scaled_Temperature[°C]"
        },
        inplace=True
    )
    out_path = os.path.join(base_path, d, "df_scaled.parquet")
    print(f"Saving: {out_path}")
    df.to_parquet(out_path, index=False)
print("Done.")
