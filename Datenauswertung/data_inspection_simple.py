import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Verzeichnis aller DataFrames
    base = Path("/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes")
    cell_dfs = {}
    for folder in base.iterdir():
        if not folder.is_dir() or not folder.name.startswith("MGFarm_18650_"):
            continue
        fp = folder / "df.parquet"
        if not fp.exists():
            continue
        # Parquet laden
        cell_dfs[folder.name] = pd.read_parquet(fp)

    # Einfacher Plot: SOH vs EFC für jede Zelle, Beschriftung direkt am Linienende
    plt.figure(figsize=(10,6))
    for name, df in cell_dfs.items():
        if "EFC" in df.columns and "SOH" in df.columns:
            x = df["EFC"].values
            y = df["SOH"].values
            plt.plot(x, y, lw=1.5)
            # Beschriftung am letzten Punkt nur mit Zellnamen
            plt.text(x[-1], y[-1],
                     name,
                     fontsize="small",
                     verticalalignment="center")

    plt.xlabel("EFC")
    plt.ylabel("SOH")
    plt.title("SOH vs EFC for all cells")
    plt.tight_layout()
    plt.savefig("SOH_vs_EFC_simple.png")
    plt.close()

if __name__ == "__main__":
    main()
