import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

# Ordner für Hyperparametertuning-Ergebnisse (HPT)
hpt_folder = Path(__file__).parent / "HPT" if '__file__' in globals() else Path("HPT")
os.makedirs(hpt_folder, exist_ok=True)

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

###############################################################################
# 5) Load best hyperparams from hpt_trials.csv
###############################################################################
csv_file = hpt_folder / "hpt_trials.csv"
df_hparams = pd.read_csv(csv_file)
best_idx = df_hparams["value"].idxmin()
best_row = df_hparams.loc[best_idx]

seq_length = int(best_row["seq_length"])
kernel_size = int(best_row["kernel_size"])
dropout = float(best_row["dropout"])
n_ch1 = int(best_row["n_ch1"])
n_ch2 = int(best_row["n_ch2"])
batch_size = 64  # override or set from best_row if stored

print(f"[INFO] Using best hyperparams for full training:\n"
      f"seq_length={seq_length}, kernel_size={kernel_size}, dropout={dropout}, "
      f"n_ch1={n_ch1}, n_ch2={n_ch2}, batch_size={batch_size}")

###############################################################################
# 6) Create train/val sets, possibly with more epochs
###############################################################################
train_dataset = SequenceDataset(df_train_scaled, seq_len=seq_length)
val_dataset   = SequenceDataset(df_val_scaled,   seq_len=seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TCN(
    input_size=2,
    num_channels=[n_ch1, n_ch2],
    kernel_size=kernel_size,
    dropout=dropout
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # or chosen from best_row

###############################################################################
# 7) Longer training with more epochs + optional early stopping
###############################################################################
n_epochs = 50
best_val_loss = float('inf')
early_stop_count = 0
PATIENCE = 5

for epoch in range(n_epochs):
    model.train()
    train_loss_sum = 0.0
    for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * x_batch.size(0)
    train_loss = train_loss_sum / len(train_loader.dataset)

    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            y_pred_val = model(x_val)
            loss_val = criterion(y_pred_val, y_val)
            val_loss_sum += loss_val.item() * x_val.size(0)
    val_loss = val_loss_sum / len(val_loader.dataset)

    print(f"[Epoch {epoch+1}/{n_epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count >= PATIENCE:
            print(f"[INFO] Early stopping at epoch {epoch+1}")
            break

###############################################################################
# 8) Save best trained model
###############################################################################
model_path = hpt_folder / "best_tcn_trained_model.pth"
if 'best_model_state' in locals():
    torch.save(best_model_state, model_path)
    print(f"[INFO] Best trained model saved at: {model_path}")
else:
    print("[WARN] No best model found during training.")