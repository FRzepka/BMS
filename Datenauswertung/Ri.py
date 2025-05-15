import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def main():
    base = Path("/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes")
    cell_dfs = {}
    for folder in sorted(base.iterdir()):
        if folder.is_dir() and folder.name.startswith("MGFarm_18650_"):
            fp = folder / "df.parquet"
            if not fp.exists():
                continue
            df = pd.read_parquet(fp)
            cell_dfs[folder.name] = df

    if not cell_dfs:
        print("Keine Zellen gefunden.")
        return

    # 1) R_i und C-Rate berechnen und Capacity sammeln
    results = []
    for name, df in cell_dfs.items():
        if "Capacity[Ah]" not in df.columns or "Schedule_Step_ID" not in df.columns:
            continue
        ids = df["Schedule_Step_ID"].values
        trans = np.where((np.roll(ids, -1)==2) & (ids==1))[0]
        if len(trans)==0 or trans[0]+2 >= len(df):
            continue
        start = trans[0] + 1
        cap = df["Capacity[Ah]"].iloc[0]
        I0 = df["Current[A]"].iloc[start]
        U0 = df["Voltage[V]"].iloc[start]
        U1 = df["Voltage[V]"].iloc[start+1]
        dU = U1 - U0

        R = dU / I0 if I0 != 0 else float("inf")
        c_rate = I0 / cap if cap != 0 else float("inf")
        # auf Viertel-Schritte runden
        c_rate_rounded = round(c_rate * 4) / 4
        results.append({
            "Cell": name,
            "Capacity[Ah]": cap,
            "Ri[Ohm]": R,
            "C_rate[1/h]": c_rate,
            "C_rate_rounded[1/h]": c_rate_rounded,
            "I0[A]": I0
        })
    pd.DataFrame(results).to_csv("Ri_results.csv", index=False)

    plt.figure(figsize=(12,6))
    colors = plt.cm.tab10.colors

    # Voltage, Current & Schedule_Step_ID vs Testtime[s], alle Zellen, Testtime startet bei 0
    for idx, (name, df) in enumerate(cell_dfs.items()):
        df0 = df.iloc[:5000].copy()
        t0 = df0["Testtime[s]"].iloc[0]
        t = df0["Testtime[s]"] - t0
        color = colors[idx % len(colors)]
        plt.plot(t, df0["Voltage[V]"],      color=color, linestyle='-',  label=f"{name} Voltage")
        plt.plot(t, df0["Current[A]"],      color=color, linestyle='--', label=f"{name} Current")
        plt.plot(t, df0["Schedule_Step_ID"],color=color, linestyle=':',  label=f"{name} Schedule_Step_ID")

    plt.xlabel("Testtime [s] (shifted to 0)")
    plt.ylabel("Value")
    plt.title("Voltage, Current & Schedule_Step_ID vs Testtime[s] (first 5000, shifted)")
    plt.legend(loc="upper right", ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig("Ri_allcells_vs_Testtime_shifted.png")
    plt.close()

    # 2) Segment ab StepID 1→2, jeweils 60 Werte, Zeit neu bei 0
    plt.figure(figsize=(12,6))
    for idx, (name, df) in enumerate(cell_dfs.items()):
        ids = df["Schedule_Step_ID"].values
        trans = np.where((np.roll(ids, -1)==2) & (ids==1))[0]
        if len(trans)==0:
            continue
        start = trans[0] + 1
        seg = df.iloc[start:start+60].copy()     # nur 60 Zeilen
        t0 = seg["Testtime[s]"].iloc[0]
        t = seg["Testtime[s]"] - t0
        color = colors[idx % len(colors)]
        plt.plot(t, seg["Voltage[V]"],      color=color, linestyle='-',  label=f"{name} V")
        plt.plot(t, seg["Current[A]"],      color=color, linestyle='--', label=f"{name} I")
        plt.plot(t, seg["Schedule_Step_ID"],color=color, linestyle=':',  label=f"{name} StepID")
        # Stromrate dI/dt berechnen und plotten
        dI = seg["Current[A]"].diff().fillna(0)
        dt_seg = seg["Testtime[s]"].diff().fillna(1)
        I_rate = dI / dt_seg
        plt.plot(t, I_rate, '-.', color=color, label=f"{name} dI/dt")

    plt.xlabel("Testtime[s] (offset at StepID 2)")
    plt.ylabel("Value")
    plt.title("Segment after StepID 1→2 (first 60 points)")
    plt.legend(loc="upper right", ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig("Ri_allcells_after_step1to2_60.png")
    plt.close()

if __name__ == "__main__":
    main()
