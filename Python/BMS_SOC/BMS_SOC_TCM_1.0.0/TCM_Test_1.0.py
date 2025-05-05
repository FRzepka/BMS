import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Ordner für Hyperparametertuning-Ergebnisse (HPT)
hpt_folder = Path(__file__).parent / "HPT" if '__file__' in globals() else Path("HPT")
os.makedirs(hpt_folder, exist_ok=True)

# Lade die CSV mit den Hyperparametern
csv_file = hpt_folder / "hpt_trials.csv"
df_hparams = pd.read_csv(csv_file)
best_idx = df_hparams["value"].idxmin()
best_row = df_hparams.loc[best_idx]

seq_length = int(best_row["seq_length"])
kernel_size = int(best_row["kernel_size"])
dropout = float(best_row["dropout"])
n_ch1 = int(best_row["n_ch1"])
n_ch2 = int(best_row["n_ch2"])

print(f"[INFO] Using best hyperparams from hpt_trials.csv:\n"
      f"seq_length={seq_length}, kernel_size={kernel_size}, dropout={dropout}, "
      f"n_ch1={n_ch1}, n_ch2={n_ch2}")

###############################################################################
# 1) Laden der Daten, Reduktion auf 25%, zeitbasierter Split in [Train, Val, Test]
###############################################################################
def load_cell_data(data_dir: Path):
    """Lade df.parquet aus dem Unterordner 'MGFarm_18650_C01'."""
    dataframes = {}
    folder = data_dir / "MGFarm_18650_C01"
    if folder.exists() and folder.is_dir():
        df_path = folder / 'df.parquet'
        if df_path.exists():
            df = pd.read_parquet(df_path)
            dataframes["C01"] = df
            print(f"[INFO] Loaded {folder.name}")
        else:
            print(f"[WARN] No df.parquet found in {folder.name}")
    else:
        print("[WARN] Folder MGFarm_18650_C01 not found")
    return dataframes

# Passe den Speicherort an
data_dir = Path('/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes')
cell_data = load_cell_data(data_dir)

cell_keys = sorted(cell_data.keys())[:1]
if len(cell_keys) < 1:
    raise ValueError("Keine Zelle gefunden; bitte prüfen.")

cell_name = cell_keys[0]  # z.B. 'C01'
df_full = cell_data[cell_name]

# Verwende 25% der Daten
sample_size = int(len(df_full) * 0.25)
df_small = df_full.head(sample_size).copy()
print(f"[INFO] Gesamtdaten: {len(df_full)}, verwende 25% = {sample_size} Zeilen.")

# Konvertiere Timestamp (Daten liegen in Sekunden)
df_small['timestamp'] = pd.to_datetime(df_small['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])

# Zeitbasiertes Split (Train: 40%, Val: 40%, Test: 20%)
len_small = len(df_small)
train_end = int(len_small * 0.4)
val_end   = int(len_small * 0.8)

df_train = df_small.iloc[:train_end]
df_val   = df_small.iloc[train_end:val_end]
df_test  = df_small.iloc[val_end:]
print(f"[INFO] Split: Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

###############################################################################
# 2) Skalierung von Voltage & Current
###############################################################################
scaler = StandardScaler()
features_to_scale = ['Voltage[V]', 'Current[A]']

# Fit nur auf Trainingsdaten
scaler.fit(df_train[features_to_scale])

df_train_scaled = df_train.copy()
df_val_scaled   = df_val.copy()
df_test_scaled  = df_test.copy()

df_train_scaled[features_to_scale] = scaler.transform(df_train_scaled[features_to_scale])
df_val_scaled[features_to_scale]   = scaler.transform(df_val_scaled[features_to_scale])
df_test_scaled[features_to_scale]  = scaler.transform(df_test_scaled[features_to_scale])

###############################################################################
# 3) Dataset-Klasse (seq2one, NICHT autoregressiv, SOC nur als Label)
###############################################################################
class SequenceDataset(Dataset):
    """
    Für jedes Sample:
      - Input: Fenster aus [Voltage, Current] über seq_len Sekunden
      - Label: SOC an (t + seq_len) (immer 1 Schritt voraus)
    """
    def __init__(self, df, seq_len=60):
        self.seq_len = seq_len
        self.features = df[["Voltage[V]", "Current[A]"]].values  # shape=(N, 2)
        self.labels   = df["SOC_ZHU"].values                     # shape=(N,)

    def __len__(self):
        return len(self.labels) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.features[idx : idx + self.seq_len]   # shape: (seq_len, 2)
        y_val = self.labels[idx + self.seq_len]             # Vorhersage: 1 Schritt
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

###############################################################################
# 4) TCN-Modell und Hilfsklassen
###############################################################################
class Chomp1d(nn.Module):
    """
    Schneidet das "causal padding" beim 1D-Convolution ab.
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x shape: (batch_size, out_channels, seq_len + padding)
        return x[:, :, :-self.chomp_size]

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        self.chomp1 = Chomp1d(chomp_size=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        self.chomp2 = Chomp1d(chomp_size=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # Residual Connection; falls in_channels != out_channels, Downsampling
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if (in_channels != out_channels) else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """
    Einfaches TCN bestehend aus einer Liste von TCN-Blöcken.
    - input_size: Anzahl der Features (hier 2: Voltage, Current)
    - num_channels: Liste der Kanalzahlen pro Block (z. B. [32, 32])
    - kernel_size und dropout sind hyperparametrierbar.
    """
    def __init__(self, input_size=2, num_channels=[32, 32], kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            # Padding = (kernel_size - 1) * dilation_size
            padding = (kernel_size - 1) * dilation_size

            block = TCNBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=padding,
                dropout=dropout
            )
            layers.append(block)

        self.tcn = nn.Sequential(*layers)
        self.fc  = nn.Linear(num_channels[-1], 1)  # finale Fully Connected-Schicht

    def forward(self, x):
        """
        x kommt als (batch_size, seq_len, input_size) rein.
        Wir permutieren zu (batch_size, input_size, seq_len), da Conv1d dies erwartet.
        """
        x = x.permute(0, 2, 1)  # => shape: (batch_size, 2, seq_len)
        y = self.tcn(x)         # => shape: (batch_size, out_channels, seq_len)
        # Nimm den letzten Zeitschritt für die 1-Schritt Vorhersage
        last_out = y[:, :, -1]
        out = self.fc(last_out)
        return out.squeeze(-1)

# Load best model file from HPT folder
best_model_path = hpt_folder / "best_tcn_soc_model.pth"

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the TCN model using the same hyperparameters we read from hpt_trials.csv.
model = TCN(
    input_size=2,
    num_channels=[n_ch1, n_ch2],
    kernel_size=kernel_size,
    dropout=dropout
)
# Load the trained weights from best_model_path.
model.load_state_dict(torch.load(best_model_path, map_location=device))
# Move the model to the chosen device and set model.eval().
model.to(device)
model.eval()

# Remove single subset usage and loop over 5 slices
n_slices = 5
slice_indices = np.linspace(0, len(df_test), n_slices + 1, dtype=int)

all_preds_list = []
all_targets_list = []
all_time_list = []
error_list = []

for i in range(n_slices):
    start_i = slice_indices[i]
    end_i = slice_indices[i + 1]
    df_test_slice = df_test.iloc[start_i:end_i].copy()
    df_test_slice_scaled = df_test_slice.copy()
    df_test_slice_scaled[features_to_scale] = scaler.transform(df_test_slice_scaled[features_to_scale])

    slice_dataset = SequenceDataset(df_test_slice_scaled, seq_len=seq_length)
    slice_loader = DataLoader(slice_dataset, batch_size=2000, shuffle=False, drop_last=True)
    
    preds_slice = []
    targets_slice = []
    with torch.no_grad():
        for x_test, y_test in slice_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_pred_test = model(x_test)
            preds_slice.append(y_pred_test.cpu().numpy())
            targets_slice.append(y_test.cpu().numpy())
    preds_slice = np.concatenate(preds_slice)
    targets_slice = np.concatenate(targets_slice)
    time_slice = df_test_slice['timestamp'].values[seq_length : seq_length + len(targets_slice)]
    # neu: Berechnung des prozentualen Fehlers
    slice_error_percent = np.mean(np.abs(preds_slice - targets_slice)) * 100
    print(f"[INFO] Slice {i+1} Error: {slice_error_percent:.2f}%")
    error_list.append(slice_error_percent)

    all_preds_list.append(preds_slice)
    all_targets_list.append(targets_slice)
    all_time_list.append(time_slice)

# Create subplots
fig, axes = plt.subplots(n_slices, 1, figsize=(12, 10))
for i, ax in enumerate(axes):
    ax.plot(all_time_list[i], all_targets_list[i], label="SOC (GT)", linestyle='-')
    ax.plot(all_time_list[i], all_preds_list[i], label="SOC (Pred)", linestyle='--')
    ax.set_title(f"Slice {i+1} | MAE: {error_list[i]:.2f}%")
    ax.set_xlabel("Time")
    ax.set_ylabel("SOC (ZHU)")
    ax.legend()
plt.tight_layout()

script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
test_dir = script_dir / "test"
test_dir.mkdir(exist_ok=True)
plot_file = test_dir / "predictions_test_slices.png"

plt.savefig(plot_file)
plt.show()
print(f"[INFO] Sliced test plot saved to: {plot_file}")