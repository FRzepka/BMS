import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # besser in reinen Terminalscreens

# Gerät auswählen und cuDNN optimieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Datenlade-Funktion
def load_cell_data(data_dir: Path):
    dataframes = {}
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("MGFarm_18650_"):
            dfp = folder / "df.parquet"
            if dfp.exists():
                dataframes[folder.name] = pd.read_parquet(dfp)
            else:
                print(f"Warning: {dfp} fehlt")
    return dataframes

# Daten vorbereiten
def load_data(base_path: str = "/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes"):
    base = Path(base_path)
    cells = load_cell_data(base)
    names = sorted(cells.keys())
    train_cells, val_cell = names[:2], names[2]

    feats = ["Voltage[V]", "Current[A]"]
    # Trainingsdaten laden und Timestamp
    train_dfs = {}
    for name in train_cells:
        df = cells[name].copy()
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        train_dfs[name] = df

    # Skalar fitten
    df_all_train = pd.concat(train_dfs.values(), ignore_index=True)
    scaler = StandardScaler().fit(df_all_train[feats])

    # Skalierte Trainingsdaten
    train_scaled = {}
    for name, df in train_dfs.items():
        df2 = df.copy()
        df2[feats] = scaler.transform(df2[feats])
        train_scaled[name] = df2

    # Validierung/Test der dritten Zelle
    df3 = cells[val_cell].copy()
    df3['timestamp'] = pd.to_datetime(df3['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    L = len(df3)
    i1, i2 = int(L*0.4), int(L*0.8)
    df_val = df3.iloc[:i1].copy()
    df_test = df3.iloc[i2:].copy()
    df_val[feats] = scaler.transform(df_val[feats])
    df_test[feats] = scaler.transform(df_test[feats])

    return train_scaled, df_val, df_test, train_cells, val_cell

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "test_plots"
    out_dir.mkdir(exist_ok=True)

    # 1) Standard‐skaliert
    train_std, df_val_std, df_test_std, train_cells, val_cell = load_data()

    # 2) Rohdaten neu einlesen und aufteilen
    base = Path("/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes")
    cells = load_cell_data(base)
    names = sorted(cells.keys())
    train_cells, val_cell = names[:2], names[2]

    train_raw = {}
    for name in train_cells:
        df = cells[name].copy()
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        train_raw[name] = df
    df3 = cells[val_cell].copy()
    df3['timestamp'] = pd.to_datetime(df3['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    L = len(df3); i1, i2 = int(L*0.4), int(L*0.8)
    df_val_raw = df3.iloc[:i1].copy()
    df_test_raw = df3.iloc[i2:].copy()

    # 3a) Voltage & SOC in [0,1], Current in [-1,1]
    feats_v = ["Voltage[V]", "SOC_ZHU"]
    feats_c = ["Current[A]"]
    # fit auf Trainingsdaten
    mm_v = MinMaxScaler(feature_range=(0,1)).fit(
        pd.concat([train_raw[n][feats_v] for n in train_cells], ignore_index=True)
    )
    mm_c = MinMaxScaler(feature_range=(-1,1)).fit(
        pd.concat([train_raw[n][feats_c] for n in train_cells], ignore_index=True)
    )
    minmax_train = {}
    for n in train_cells:
        df = train_raw[n].copy()
        df[feats_v] = mm_v.transform(df[feats_v])
        df[feats_c] = mm_c.transform(df[feats_c])
        minmax_train[n] = df
    df_val_mm = df_val_raw.copy()
    df_val_mm[feats_v] = mm_v.transform(df_val_raw[feats_v])
    df_val_mm[feats_c] = mm_c.transform(df_val_raw[feats_c])
    df_test_mm = df_test_raw.copy()
    df_test_mm[feats_v] = mm_v.transform(df_test_raw[feats_v])
    df_test_mm[feats_c] = mm_c.transform(df_test_raw[feats_c])

    # 4) Print columns + erste Zeile
    for n in train_cells:
        print(f"\n=== Train Cell {n} ===")
        print(" raw cols:", train_raw[n].columns.tolist())
        print(train_raw[n].head(1))
        print(" std cols:", train_std[n].columns.tolist())
        print(train_std[n].head(1))
        print(" mm cols:", minmax_train[n].columns.tolist())
        print(minmax_train[n].head(1))
    for label, df_raw, df_std, df_mm in [
        ("Val", df_val_raw, df_val_std, df_val_mm),
        ("Test", df_test_raw, df_test_std, df_test_mm)
    ]:
        print(f"\n=== {label} Cell {val_cell} ===")
        print(" raw cols:", df_raw.columns.tolist()); print(df_raw.head(1))
        print(" std cols:", df_std.columns.tolist()); print(df_std.head(1))
        print(" mm cols:", df_mm.columns.tolist()); print(df_mm.head(1))

    # 5) Plots: für jede Zell‐ und Skalierungs‐Version
    def plot_three(df, title, fname):
        fig, axs = plt.subplots(3,1,figsize=(10,8), sharex=True)
        axs[0].plot(df['timestamp'], df['Voltage[V]']); axs[0].set_ylabel('Voltage')
        axs[1].plot(df['timestamp'], df['Current[A]'], color='C1'); axs[1].set_ylabel('Current')
        axs[2].plot(df['timestamp'], df['SOC_ZHU'], color='k'); axs[2].set_ylabel('SOC'); axs[2].set_xlabel('Time')
        fig.suptitle(title)
        fig.tight_layout(rect=[0,0,1,0.95])
        fig.savefig(out_dir / fname)
        plt.close(fig)

    # Train
    for n in train_cells:
        plot_three(train_raw[n],  f"Train orig  {n}", f"train_{n}_orig.png")
        plot_three(train_std[n],  f"Train std   {n}", f"train_{n}_std.png")
        plot_three(minmax_train[n], f"Train mm    {n}", f"train_{n}_mm.png")
    # Val & Test
    plot_three(df_val_raw,  f"Val orig   {val_cell}", f"val_{val_cell}_orig.png")
    plot_three(df_val_std,  f"Val std    {val_cell}", f"val_{val_cell}_std.png")
    plot_three(df_val_mm,   f"Val mm     {val_cell}", f"val_{val_cell}_mm.png")

    plot_three(df_test_raw, f"Test orig  {val_cell}", f"test_{val_cell}_orig.png")
    plot_three(df_test_std, f"Test std   {val_cell}", f"test_{val_cell}_std.png")
    plot_three(df_test_mm,  f"Test mm    {val_cell}", f"test_{val_cell}_mm.png")
