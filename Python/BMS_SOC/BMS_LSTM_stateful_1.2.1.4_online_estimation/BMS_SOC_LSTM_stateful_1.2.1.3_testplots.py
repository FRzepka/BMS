import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import gc

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

if __name__ == "__main__":
    print("== Starting 1.2.1.3 testplots debug ==")
    base = Path("/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes")
    cells = load_cell_data(base)
    print("Cells found:", list(cells.keys()))

    out_dir = Path(__file__).parent / "test_plots"
    out_dir.mkdir(exist_ok=True)

    # 1) Globales Zusammenfügen zum Fitten der Scaler
    dfs = []
    for name, df0 in cells.items():
        print(f"Load for scaler fit: {name}, rows={len(df0)}")
        df = df0.copy()
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    print("Concatenated total rows for scaler:", len(df_all))

    # 2) Fit StandardScaler auf Voltage & Current
    feats_std = ["Voltage[V]", "Current[A]"]
    scaler_std = StandardScaler().fit(df_all[feats_std])
    print("StandardScaler fitted:", feats_std)

    # 3) Fit MinMaxScaler: Voltage & SOC [0,1], Current [-1,1]
    feats_v = ["Voltage[V]", "SOC_ZHU"]
    feats_c = ["Current[A]"]
    mm_v = MinMaxScaler(feature_range=(0,1)).fit(df_all[feats_v])
    mm_c = MinMaxScaler(feature_range=(-1,1)).fit(df_all[feats_c])
    print("MinMaxScalers fitted for", feats_v, feats_c)

    # Speicher freigeben
    del df_all, dfs; gc.collect()

    # 4) Plot-Funktion
    def plot_three(df, name, version):
        fig, axs = plt.subplots(3,1,figsize=(10,8), sharex=True)
        axs[0].plot(df['timestamp'], df['Voltage[V]']); axs[0].set_ylabel('Voltage')
        axs[1].plot(df['timestamp'], df['Current[A]'], color='C1'); axs[1].set_ylabel('Current')
        axs[2].plot(df['timestamp'], df['SOC_ZHU'], color='k'); axs[2].set_ylabel('SOC')
        axs[2].set_xlabel('Time')
        fig.suptitle(f"{version} {name}")
        fig.tight_layout(rect=[0,0,1,0.95])
        fig.savefig(out_dir / f"{version.lower()}_{name}.png")
        plt.close(fig)

    # 5) Zellweise Verarbeiten: raw → std → mm → plot → löschen
    for name, df0 in cells.items():
        print(f"\nProcessing cell {name}")
        df = df0.copy()
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        print(" raw rows:", len(df))

        # raw plot
        plot_three(df, name, "orig")

        # standard scale
        df_std = df.copy()
        df_std[feats_std] = scaler_std.transform(df_std[feats_std])
        plot_three(df_std, name, "std")
        del df_std

        # minmax scale
        df_mm = df.copy()
        df_mm[feats_v] = mm_v.transform(df_mm[feats_v])
        df_mm[feats_c] = mm_c.transform(df_mm[feats_c])
        plot_three(df_mm, name, "mm")
        del df_mm, df
        gc.collect()
        print(f"Completed {name}")

    print("== All cells processed ==")
