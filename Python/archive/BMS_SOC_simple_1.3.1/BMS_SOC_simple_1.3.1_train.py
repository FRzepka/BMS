import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_forecasting.models.nn.rnn import LSTM as ForecastingLSTM
from tqdm import tqdm  # neuen Import hinzufügen

def find_cell_folders(data_dir: Path, pattern=r"MGFarm_18650_C\d+"):
    cell_folders = []
    for item in data_dir.iterdir():
        if item.is_dir() and re.match(pattern, item.name):
            cell_folders.append(item)
    return sorted(cell_folders)

def load_cell_data(folder: Path):
    df_path = folder / "df_scaled.parquet"
    if not df_path.exists():
        print(f"[WARN] Datei {df_path} nicht gefunden.")
        return None
    return pd.read_parquet(df_path)

def to_input_array(df):
    return df[["Scaled_Voltage[V]", "Scaled_Current[A]", "SOC_ZHU"]].values

# ---------------- GPU Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
# Daten laden, Zellenordner finden und Trainings-/Testdaten splitten
###############################################################################
data_dir = Path('/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes')
all_cell_folders = find_cell_folders(data_dir)
if len(all_cell_folders) == 0:
    raise ValueError("Keine passenden Zellenordner gefunden!")

cell_data_dict = {}
all_trainval_dfs = []

for folder in all_cell_folders:
    df_full = load_cell_data(folder)
    if df_full is None:
        continue
    cell_name = folder.name.split("_")[-1]
    sample_size = int(len(df_full) * 0.1)
    df_train_val = df_full.head(sample_size)
    mid_start = (len(df_full) - sample_size) // 2
    df_test = df_full.iloc[mid_start : mid_start + sample_size]
    df_train_val['timestamp'] = pd.to_datetime(df_train_val['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    df_test['timestamp'] = pd.to_datetime(df_test['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    df_train_val_orig = df_train_val.copy()
    train_cut = int(len(df_train_val_orig) * 0.8)
    train_data_orig = df_train_val_orig.iloc[:train_cut].reset_index(drop=True)
    val_data_orig   = df_train_val_orig.iloc[train_cut:].reset_index(drop=True)
    cell_data_dict[cell_name] = {"train_data_orig": train_data_orig, "val_data_orig": val_data_orig}
    combined_trainval = pd.concat([train_data_orig, val_data_orig], axis=0)
    all_trainval_dfs.append(combined_trainval)

if len(all_trainval_dfs) == 0:
    raise ValueError("Es wurden zwar Zellenordner gefunden, aber keine df_scaled.parquet.")

df_trainval_full = pd.concat(all_trainval_dfs, axis=0).reset_index(drop=True)
trainval_array = to_input_array(df_trainval_full)
print(f"Train/Val Gesamt (alle Zellen): {len(df_trainval_full)} Zeilen.")

###############################################################################
# Dataset-Definition
###############################################################################
seq_length = 60

class SequenceDataset(Dataset):
    def __init__(self, data_array, seq_len=60):
        self.data_array = data_array
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data_array) - self.seq_len
    def __getitem__(self, idx):
        x_seq = self.data_array[idx : idx + self.seq_len]
        y_val = self.data_array[idx + self.seq_len, 2]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

trainval_dataset = SequenceDataset(trainval_array, seq_len=seq_length)
trainval_loader = DataLoader(trainval_dataset, batch_size=32, shuffle=True)

###############################################################################
# Modell-Definition mittels ForecastingLSTM
###############################################################################
class LSTMSOCModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=1, batch_first=True):
        super().__init__()
        self.lstm = ForecastingLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        soc_pred = self.fc(last_out)
        return soc_pred.squeeze(-1)

model = LSTMSOCModel(input_size=3, hidden_size=32, num_layers=1, batch_first=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

###############################################################################
# Trainings Loop mit Epochen- und Batchfortschritt
###############################################################################
epochs = 20
model.train()

for epoch in range(epochs):
    train_losses = []
    total_batches = len(trainval_loader)
    print(f"Starte Epoche {epoch+1}/{epochs}")
    # tqdm umschließt den DataLoader (Anzeige in Prozent)
    for x_batch, y_batch in tqdm(trainval_loader, desc=f"Epoche {epoch+1}/{epochs}", unit="batch", leave=False):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    print(f"Epoche {epoch+1}/{epochs} abgeschlossen - Durchschnittlicher Verlust: {np.mean(train_losses):.6f}")

###############################################################################
# Modell speichern im Ordner "models" relativ zum Script-Verzeichnis
###############################################################################
current_dir = Path(__file__).parent
models_dir = current_dir / "models"
os.makedirs(models_dir, exist_ok=True)
model_path = models_dir / "lstm_soc_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Modell gespeichert unter: {model_path}")
