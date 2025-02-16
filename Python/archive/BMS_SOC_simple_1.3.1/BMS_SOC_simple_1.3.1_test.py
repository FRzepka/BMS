import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_forecasting.models.nn.rnn import LSTM as ForecastingLSTM

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

# Dataset- und Modell-Definition
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

def predict_autoregressive(model, data_array, seq_len=60):
    model.eval()
    data_clone = data_array.copy()
    preds = np.full(len(data_clone), np.nan)
    with torch.no_grad():
        for i in range(seq_len, len(data_clone)):
            input_seq = data_clone[i-seq_len : i]
            x_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred_soc = model(x_t).item()
            preds[i] = pred_soc
            data_clone[i, 2] = pred_soc
    return preds

# Testdaten laden und Modell laden aus "models"
data_dir = Path('/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes')
all_cell_folders = find_cell_folders(data_dir)
if len(all_cell_folders) == 0:
    raise ValueError("Keine passenden Zellenordner gefunden!")

cell_data_dict = {}
for folder in all_cell_folders:
    df_full = load_cell_data(folder)
    if df_full is None:
        continue
    cell_name = folder.name.split("_")[-1]
    sample_size = int(len(df_full) * 0.001)
    mid_start = (len(df_full) - sample_size) // 2
    df_test = df_full.iloc[mid_start : mid_start + sample_size]
    df_test['timestamp'] = pd.to_datetime(df_test['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    cell_data_dict[cell_name] = {"test_data_orig": df_test}

# Modell laden aus Ordner "models"
current_dir = Path(__file__).parent
models_dir = current_dir / "models"
model_path = models_dir / "lstm_soc_model.pth"
if not model_path.exists():
    raise FileNotFoundError(f"Modell nicht gefunden unter: {model_path}")

model = LSTMSOCModel(input_size=3, hidden_size=32, num_layers=1, batch_first=True).to(device)
model.load_state_dict(torch.load(model_path))
print(f"Modell geladen aus: {model_path}")

# Test & Plot der Vorhersagen pro Zelle, Plots werden im Ordner "models" gespeichert
for cell_name, data_dict in cell_data_dict.items():
    df_test_orig = data_dict["test_data_orig"]
    test_array = to_input_array(df_test_orig)
    preds_test = predict_autoregressive(model, test_array, seq_len=seq_length)
    time_test = df_test_orig['timestamp'].values
    gt_test   = test_array[:, 2]
    
    plt.figure(figsize=(12, 5))
    plt.plot(time_test, gt_test, label=f"Ground Truth SOC ({cell_name})", color='k')
    plt.plot(time_test, preds_test, label="Predicted SOC (autoregressive)", color='r', alpha=0.7)
    plt.title(f"Autoregressive SOC-Vorhersage - Testblock für Zelle {cell_name}")
    plt.xlabel("Time")
    plt.ylabel("SOC")
    plt.legend()
    plt.tight_layout()
    
    plot_file = models_dir / f"{cell_name}_prediction.png"
    plt.savefig(plot_file)
    print(f"Plot für Zelle {cell_name} gespeichert unter: {plot_file}")
    plt.show()
