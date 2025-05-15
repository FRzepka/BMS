import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D

def main():
    # --- Global font & style settings ---
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

    # --- Load all cell dataframes ---
    base = Path("/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes")
    cell_dfs = {}
    for folder in sorted(base.iterdir()):
        if folder.is_dir() and folder.name.startswith("MGFarm_18650_"):
            fp = folder / "df.parquet"
            if not fp.exists():
                continue
            df = pd.read_parquet(fp)
            if "timestamp" not in df.columns:
                df["timestamp"] = pd.to_datetime(df["Absolute_Time[yyyy-mm-dd hh:mm:ss]"])
            cell_dfs[folder.name] = df

    print("Loaded cells:", list(cell_dfs.keys()))
    if cell_dfs:
        name, df0 = next(iter(cell_dfs.items()))
        print(f"Columns of {name}:", df0.columns.tolist())

    # --- 1) SOH vs Time of all cells (unchanged) ---
    plt.figure(figsize=(10,6))
    for name, df in cell_dfs.items():
        if "SOH" in df.columns:
            plt.plot(df["timestamp"], df["SOH"], label=name)
    plt.xlabel("Time")
    plt.ylabel("SOH")
    plt.title("SOH vs Time of all cells")
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig("SOH_vs_Time.png")
    plt.close()

    # --- 2) SOH vs EFC with blue spectrum and concise legend ---
    meta = {
        'MGFarm_18650_C01': {'C_rate_dchg': 1.0,    'DOD': 45},
        'MGFarm_18650_C02': {'C_rate_dchg': 1.0,    'DOD': 45},
        'MGFarm_18650_C03': {'C_rate_dchg': 1.0,    'DOD': 65},
        'MGFarm_18650_C04': {'C_rate_dchg': 1.0,    'DOD': 65},
        'MGFarm_18650_C05': {'C_rate_dchg': 1.0,    'DOD': 45},
        'MGFarm_18650_C06': {'C_rate_dchg': 1.0,    'DOD': 45},
        'MGFarm_18650_C07': {'C_rate_dchg': 1.0,    'DOD': 65},
        'MGFarm_18650_C08': {'C_rate_dchg': 1.0,    'DOD': 65},
        'MGFarm_18650_C09': {'C_rate_dchg': 2.5,    'DOD': 45},
        'MGFarm_18650_C10': {'C_rate_dchg': 2.5,    'DOD': 45},
        'MGFarm_18650_C11': {'C_rate_dchg': 2.5,    'DOD': 65},
        'MGFarm_18650_C12': {'C_rate_dchg': 2.5,    'DOD': 65},
        'MGFarm_18650_C13': {'C_rate_dchg': 2.5,    'DOD': 45},
        'MGFarm_18650_C14': {'C_rate_dchg': 2.5,    'DOD': 45},
        'MGFarm_18650_C15': {'C_rate_dchg': 2.5,    'DOD': 65},
        'MGFarm_18650_C16': {'C_rate_dchg': 2.5,    'DOD': 65},
        'MGFarm_18650_C17': {'C_rate_dchg': 1.75,   'DOD': 55},
        'MGFarm_18650_C18': {'C_rate_dchg': 1.75,   'DOD': 55},
        'MGFarm_18650_C19': {'C_rate_dchg': 1.75,   'DOD': 38.18},
        'MGFarm_18650_C20': {'C_rate_dchg': 1.75,   'DOD': 38.18},
        'MGFarm_18650_C21': {'C_rate_dchg': 1.75,   'DOD': 71.82},
        'MGFarm_18650_C22': {'C_rate_dchg': 1.75,   'DOD': 71.82},
        'MGFarm_18650_C23': {'C_rate_dchg': 1.75,   'DOD': 55},
        'MGFarm_18650_C24': {'C_rate_dchg': 1.75,   'DOD': 55},
        'MGFarm_18650_C25': {'C_rate_dchg': 1.75,   'DOD': 55},
        'MGFarm_18650_C26': {'C_rate_dchg': 1.75,   'DOD': 55},
        'MGFarm_18650_C27': {'C_rate_dchg': 0.4887, 'DOD': 55},
        'MGFarm_18650_C28': {'C_rate_dchg': 0.4887, 'DOD': 55},
        'MGFarm_18650_C29': {'C_rate_dchg': 3.0113, 'DOD': 55},
        'MGFarm_18650_C30': {'C_rate_dchg': 3.0113, 'DOD': 55},
    }

    rates = sorted({v['C_rate_dchg'] for v in meta.values()})
    norm = plt.Normalize(vmin=min(rates), vmax=max(rates))
    cmap = plt.cm.Blues  # light→dark blue

    fig, ax = plt.subplots(figsize=(10,6))
    # plot in ascending C-rate order
    for name in sorted(cell_dfs,
                       key=lambda n: meta.get(n,{}).get('C_rate_dchg',0)):
        if name not in meta:
            continue
        df = cell_dfs[name]
        cr  = meta[name]['C_rate_dchg']
        dod = meta[name]['DOD']
        ax.plot(
            df["EFC"], df["SOH"],
            color=cmap(norm(cr)), lw=1.5,
            label=f"{cr:.1f}C, {dod:.0f}%"
        )

    ax.set_xlabel("EFC")
    ax.set_ylabel("SOH")
    ax.set_title("SOH vs EFC of all cells")
    # Legend mit jedem Eintrag (C-rate & DOD)
    leg = ax.legend(title="C-rate   DOD",
                    loc="lower left", frameon=True)
    frame = leg.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('gray')

    plt.tight_layout()
    plt.savefig("SOH_vs_EFC_colormap.png")
    plt.close()

    # --- Extra plot: SOH vs EFC with cell names for checking ---
    fig, ax = plt.subplots(figsize=(10,6))
    # plot with cell names, sorted by C-rate
    for name in sorted(cell_dfs,
                       key=lambda n: meta.get(n,{}).get('C_rate_dchg',0)):
        if name not in meta:
            continue
        df = cell_dfs[name]
        cr = meta[name]['C_rate_dchg']
        ax.plot(
            df["EFC"], df["SOH"],
            color=cmap(norm(cr)), lw=1.5,
            label=name
        )
    ax.set_xlabel("EFC")
    ax.set_ylabel("SOH")
    ax.set_title("SOH vs EFC of all cells (with cell names)")
    leg = ax.legend(loc="lower left", frameon=True)
    frame = leg.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('gray')
    plt.tight_layout()
    plt.savefig("SOH_vs_EFC_cells.png")
    plt.close()

    # --- 3) Capacity distribution boxplot (unchanged) ---
    capacities = [df["Capacity[Ah]"].iloc[0] for df in cell_dfs.values()
                  if "Capacity[Ah]" in df.columns]
    plt.figure(figsize=(6,4))
    plt.boxplot(
        capacities, vert=False, patch_artist=True,
        tick_labels=["LFP_18650"], showmeans=True
    )
    plt.xlabel("Capacity [Ah]")
    plt.title("Capacity Distribution (LFP_18650)")
    plt.tight_layout()
    plt.savefig("capacity_boxplot.png")
    plt.close()

    # --- 4) Ri distribution boxplot (unchanged) ---
    df_ri = pd.read_csv("Ri_results.csv")
    ri_vals = df_ri.loc[df_ri["C_rate_rounded[1/h]"] == -1.0, "Ri[Ohm]"].values
    plt.figure(figsize=(6,4))
    plt.boxplot(
        ri_vals, vert=False, patch_artist=True,
        tick_labels=["Rᵢ @ 100% SOC, 1 C"], showmeans=True
    )
    plt.xlabel("Ri [Ω]")
    plt.title("Ri Distribution at 100% SOC and 1 C Discharge")
    plt.tight_layout()
    plt.savefig("Ri_boxplot.png")
    plt.close()

    # --- Combined Boxplots (only moving titles to x-labels) ---
    ri_m = ri_vals * 1000  # in mΩ
    fig, axes = plt.subplots(1, 2, figsize=(12,6), sharey=False)
    medianprops  = dict(color='black', linewidth=2)
    meanprops    = dict(marker='D', markeredgecolor='black',
                        markerfacecolor='white', markersize=8)
    boxprops     = dict(facecolor='lightblue', edgecolor='black')
    whiskerprops = dict(color='black', linewidth=1)
    capprops     = dict(color='black', linewidth=1)
    legend_labels = ['Median', 'Mean', 'Box = 25–75%', 'Whiskers']

    # Capacity subplot
    ax = axes[0]
    parts_cap = ax.boxplot(
        capacities, vert=True, patch_artist=True, showmeans=True,
        medianprops=medianprops, meanprops=meanprops,
        boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops
    )
    ax.set_ylabel("Capacity [Ah]")
    ax.set_xticks([])
    ax.set_xlabel(f"Initial Capacity at 100 % SOH (n={len(capacities)})")
    m = np.median(capacities)
    iq = np.percentile(capacities, 75) - np.percentile(capacities, 25)
    ax.annotate(f"IQR: {iq:.3f} Ah ({iq/m*100:.1f} %)",
                xy=(0.05,0.95), xycoords='axes fraction', va='top')
    handles = [parts_cap['medians'][0], parts_cap['means'][0],
               parts_cap['boxes'][0], parts_cap['whiskers'][0]]
    ax.legend(handles, legend_labels,
              loc='upper right', frameon=False, title='Elements')

    # Ri subplot
    ax = axes[1]
    parts_ri = ax.boxplot(
        ri_m, vert=True, patch_artist=True, showmeans=True,
        medianprops=medianprops, meanprops=meanprops,
        boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops
    )
    ax.set_ylabel("Internal Resistance [mΩ]")
    ax.set_xticks([])
    ax.set_xlabel(f"Initial Internal Resistance at 1 C, 100 % SOC (n={len(ri_m)})")
    m2 = np.median(ri_m)
    iq2 = np.percentile(ri_m, 75) - np.percentile(ri_m, 25)
    ax.annotate(f"IQR: {iq2:.1f} mΩ ({iq2/m2*100:.1f} %)",
                xy=(0.05,0.95), xycoords='axes fraction', va='top')
    handles2 = [parts_ri['medians'][0], parts_ri['means'][0],
                parts_ri['boxes'][0], parts_ri['whiskers'][0]]
    ax.legend(handles2, legend_labels,
              loc='upper right', frameon=False, title='Elements')

    plt.suptitle(
        "Initial Capacity & Initial Internal Resistance Distributions\n"
        "for JGCFR18650-1800 mAh LFP Cells"
    )
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig("combined_boxplots_final.png", bbox_inches='tight')
    plt.close()

    # --- 5) Q_m und Q_c über Absolute Time plotten (nur letzte Zelle) ---
    if cell_dfs:
        rest_name = list(cell_dfs.keys())[-1]
        df = cell_dfs[rest_name]
        print(f"Plotting Q_m and Q_c for cell: {rest_name}")
        if all(col in df.columns for col in ["Q_m", "Q_c"]):
            plt.figure(figsize=(10,6))
            t = pd.to_datetime(df["Absolute_Time[yyyy-mm-dd hh:mm:ss]"])
            plt.plot(t, df["Q_m"], label=f"{rest_name} Q_m")
            plt.plot(t, df["Q_c"], label=f"{rest_name} Q_c")
            plt.xlabel("Time")
            plt.ylabel("Charge")
            plt.title(f"Q_m and Q_c over Time for {rest_name}")
            print(f"Saving Q_metrics_over_time_{rest_name}.png")
            plt.legend(loc="best", frameon=False)
            plt.tight_layout()
            plt.savefig(f"Q_metrics_over_time_{rest_name}.png")
            plt.close()

if __name__ == "__main__":
    main()
