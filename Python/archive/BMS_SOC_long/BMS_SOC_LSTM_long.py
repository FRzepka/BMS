import os
import sys
import random
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Device-Auswahl
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
# 1) Laden der Daten
###############################################################################
def load_all_cell_data(data_dir: Path):
    """
    Durchsucht das Verzeichnis `data_dir` nach allen Unterordnern,
    die mit 'MGFarm_18650_C' beginnen. In jedem solchen Ordner wird
    die Datei 'df_scaled.parquet' eingelesen (falls vorhanden).

    Gibt ein Dict zurück:
      {
         "C01": DataFrame,  # aus Ordner MGFarm_18650_C01
         "C02": DataFrame,  # aus Ordner MGFarm_18650_C02
         ...
      }
    """
    dataframes = {}
    for folder in data_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("MGFarm_18650_C"):
            cell_name = folder.name.replace("MGFarm_18650_", "")
            df_path = folder / 'df_scaled.parquet'  # Lade df_scaled.parquet
            if df_path.exists():
                df = pd.read_parquet(df_path)
                dataframes[cell_name] = df
                print(f"Loaded: {folder.name} -> Key: {cell_name}")
            else:
                print(f"Warning: No df_scaled.parquet found in {folder.name}")
    return dataframes

data_dir = Path('/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes')
cell_data = load_all_cell_data(data_dir)

# Sicherstellen, dass es mindestens 4 Zellen gibt:
if len(cell_data) < 4:
    raise ValueError(
        f"Es wurden nur {len(cell_data)} Zellen gefunden. "
        "Bitte sicherstellen, dass mindestens 4 Zellen vorhanden sind."
    )

# Genau 4 Zellen auswählen (z.B. alphabetisch sortiert, dann die ersten 4 nehmen):
all_cells = sorted(cell_data.keys())
selected_cells = all_cells[:4]
print("Verwendete Zellen:", selected_cells)

# Reduziere cell_data auf die 4 gewählten Zellen
cell_data_4 = {c: cell_data[c] for c in selected_cells}

###############################################################################
# 2) Zellen für Training, Validierung und Test auswählen
###############################################################################
train_cells = selected_cells[:2]
val_cell = selected_cells[2]
test_cell = selected_cells[3]

print(f"Train-Zellen: {train_cells}")
print(f"Val-Zelle: {val_cell}")
print(f"Test-Zelle: {test_cell}")

###############################################################################
# 3) Vorbereitung der Daten (keine Skalierung mehr notwendig)
###############################################################################
# (a) Timestamp-Spalte in Datetime umwandeln (falls noch nicht geschehen)
for cell_name in tqdm(cell_data_4, desc="Preparing timestamps"):
    cell_data_4[cell_name]['timestamp'] = pd.to_datetime(
        cell_data_4[cell_name]['Absolute_Time[yyyy-mm-dd hh:mm:ss]'],
        errors='coerce'
    )

# (b) Skaliertes Dict erstellen (einfach die vorhandenen DataFrames verwenden)
scaled_data = cell_data_4

###############################################################################
# 4) Dataset-Klasse & Modell
###############################################################################
class SequenceDataset(Dataset):
    """
    Gibt (seq, next_value) zurück, d.h.
      X[t] = (Voltage[t], Current[t], SOC[t])  (über seq_length Timesteps)
      y[t] = SOC[t+1].
    """
    def __init__(self, data_array, seq_len=60):
        self.data_array = data_array
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data_array) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.data_array[idx : idx + self.seq_len, :3]  # shape (seq_len, 3)
        y_val = self.data_array[idx + self.seq_len, 3]     # Spalte 3 = SOC
        x_seq_t = torch.tensor(x_seq, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        return x_seq_t, y_val_t


class LSTMSOCModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x.shape = (batch_size, seq_length, input_size=3)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # letzter Zeitschritt im Output:
        last_out = lstm_out[:, -1, :]  # shape = (batch_size, hidden_size)
        soc_pred = self.fc(last_out)   # shape = (batch_size, 1)
        return soc_pred.squeeze(-1)    # -> (batch_size,)


def predict_autoregressive(model, data_array, seq_len=60):
    """
    Autoregressive Vorhersage:
    Nutzt das erste seq_len-Fenster mit echten SOC-Werten,
    ab Schritt seq_len wird SOC[t+1] durch Modell ersetzt.
    """
    model.eval()
    data_clone = data_array.copy()
    preds = np.full(len(data_clone), np.nan)

    with torch.no_grad():
        for i in tqdm(range(seq_len, len(data_clone)), desc="Prediction"):
            input_seq = data_clone[i-seq_len : i]  # (seq_len, 3)
            x_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred_soc = model(x_t).item()
            preds[i] = pred_soc
            # Autoregressiv: SOC-Spalte überschreiben
            data_clone[i, 2] = pred_soc

    return preds

###############################################################################
# 5) Training
###############################################################################
seq_length = 60
EPOCHS = 20

# Modell und Optimizer initialisieren
model = LSTMSOCModel(input_size=3, hidden_size=32, num_layers=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Daten für Training und Validierung vorbereiten
df_train_list = []
for c in train_cells:
    df_train_list.append(scaled_data[c][["Scaled_Voltage[V]", "Scaled_Current[A]", "Scaled_Temperature[°C]", "SOC_ZHU"]])
df_train_all = pd.concat(df_train_list, ignore_index=True)
df_val = scaled_data[val_cell][["Scaled_Voltage[V]", "Scaled_Current[A]", "Scaled_Temperature[°C]", "SOC_ZHU"]].copy()

train_array = df_train_all.values
val_array = df_val.values

train_dataset = SequenceDataset(train_array, seq_len=seq_length)
val_dataset = SequenceDataset(val_array, seq_len=seq_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training Loop
train_losses_per_epoch = []
val_losses_per_epoch = []

print("\n=== Training ===")
for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
    model.train()
    train_losses = []
    for x_batch, y_batch in tqdm(train_loader, desc="Training"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Validierung am Ende jeder Epoche
    val_losses = []
    model.eval()
    with torch.no_grad():
        for x_val, y_val in tqdm(val_loader, desc="Validation"):
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            y_pred_val = model(x_val)
            loss_val = criterion(y_pred_val, y_val)
            val_losses.append(loss_val.item())

    train_loss_mean = np.mean(train_losses)
    val_loss_mean = np.mean(val_losses)
    train_losses_per_epoch.append(train_loss_mean)
    val_losses_per_epoch.append(val_loss_mean)

    print(f"  Epoch [{epoch}/{EPOCHS}] - "
          f"Train MSE: {train_loss_mean:.4f} | "
          f"Val MSE: {val_loss_mean:.4f}")

# Speichere das trainierte Modell
torch.save(model.state_dict(), "lstm_soc_model.pth")
print("Trained model saved as lstm_soc_model.pth")

# Plot der Validierungsverluste
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), val_losses_per_epoch, label="Validation Loss")
plt.title("Validation Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

print("Fertig! Training abgeschlossen.")
