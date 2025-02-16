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

# Falls noch nicht installiert:
# pip install pytorch-forecasting
from pytorch_forecasting.models.nn.rnn import LSTM as ForecastingLSTM

###############################################################################
# 1) Alle passenden Zellen-Unterordner finden, df_scaled.parquet laden
###############################################################################
def find_cell_folders(data_dir: Path, pattern=r"MGFarm_18650_C\d+"):
    """
    Durchsucht data_dir nach Unterordnern, deren Name zu pattern passt
    (z.B. MGFarm_18650_C01, MGFarm_18650_C03, ...).
    Gibt eine sortierte Liste dieser Ordner zurück.
    """
    cell_folders = []
    for item in data_dir.iterdir():
        if item.is_dir() and re.match(pattern, item.name):
            cell_folders.append(item)
    cell_folders = sorted(cell_folders)
    return cell_folders

def load_cell_data(folder: Path):
    """
    Lädt df_scaled.parquet aus dem angegebenen Ordner.
    Gibt ein DataFrame zurück oder None, falls die Datei fehlt.
    """
    df_path = folder / "df_scaled.parquet"
    if not df_path.exists():
        print(f"[WARN] Datei {df_path} nicht gefunden.")
        return None
    df = pd.read_parquet(df_path)
    return df

###############################################################################
# 2) Pfad konfigurieren & nur 2 Zellen laden
###############################################################################
data_dir = Path('/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes')
all_cell_folders = find_cell_folders(data_dir)  # z.B. C01, C02, C03, ...
if len(all_cell_folders) == 0:
    raise ValueError("Keine passenden Zellenordner gefunden!")
    
# Nur die ersten 2 verwenden
selected_folders = all_cell_folders[:2]
print("Gefundene Zellen:", [f.name for f in selected_folders])

###############################################################################
# 3) 0.1% Train/Val, 0.1% Test aus der Mitte – für jede Zelle
#    Dann Train/Val-Daten zusammenführen und aus Zelle #2 den Test verwenden.
###############################################################################
cell_data_list = []  # Hier sammeln wir pro Zelle: train_data, val_data, test_data, test_data_orig, ...
cell_names = []

for folder in selected_folders:
    df_full = load_cell_data(folder)
    if df_full is None:
        continue
    
    # Zellenname extrahieren, z.B. "C01" aus "MGFarm_18650_C01"
    cell_name = folder.name.split("_")[-1]  # "C01"
    cell_names.append(cell_name)
    
    # 0.1% der Daten als Train/Val
    sample_size = int(len(df_full) * 0.1)
    df_train_val = df_full.head(sample_size)
    
    # Ein mittlerer Block (ebenfalls 0.1%) als Test
    mid_start = (len(df_full) - sample_size) // 2
    df_test = df_full.iloc[mid_start : mid_start + sample_size]
    
    # Timestamp konvertieren
    df_train_val.loc[:, 'timestamp'] = pd.to_datetime(df_train_val['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    df_test.loc[:, 'timestamp'] = pd.to_datetime(df_test['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    
    # Originale Kopien für Plots etc.
    df_train_val_orig = df_train_val.copy()
    df_test_orig = df_test.copy()
    
    # Train/Val Split (80/20)
    total = len(df_train_val)
    train_cut = int(total * 0.8)
    
    train_data_orig = df_train_val_orig.iloc[:train_cut].reset_index(drop=True)
    val_data_orig   = df_train_val_orig.iloc[train_cut:].reset_index(drop=True)
    test_data_orig  = df_test_orig.reset_index(drop=True)
    
    # In diesem Beispiel haben wir ja schon "Scaled_Voltage[V]", "Scaled_Current[A]", "SOC_ZHU"
    # => Keine weitere Skalierung nötig.
    
    # Speichern für nachher
    cell_data_list.append({
        "cell_name": cell_name,
        "train_data_orig": train_data_orig,
        "val_data_orig":   val_data_orig,
        "test_data_orig":  test_data_orig,
    })

# Prüfen, ob wir wirklich 2 Zellen haben
if len(cell_data_list) < 2:
    raise ValueError(f"Es wurden weniger als 2 Zellen geladen! Gefunden: {len(cell_data_list)}")

###############################################################################
# 4) Train/Val-Daten beider Zellen zusammenführen
#    Wir packen die skalierten Spalten in numpy-Arrays
###############################################################################
def to_input_array(df):
    # ACHTUNG: hier die Spaltennamen anpassen, falls sie abweichen
    return df[["Scaled_Voltage[V]", "Scaled_Current[A]", "SOC_ZHU"]].values

# Cell 1
cell1_train = cell_data_list[0]["train_data_orig"]
cell1_val   = cell_data_list[0]["val_data_orig"]
# Cell 2
cell2_train = cell_data_list[1]["train_data_orig"]
cell2_val   = cell_data_list[1]["val_data_orig"]

# Wir bauen einfach: trainval_df = alle train + val Zeilen (Cell1 + Cell2)
df_trainval = pd.concat([cell1_train, cell1_val, cell2_train, cell2_val], axis=0).reset_index(drop=True)

# Nimm als Test-Set einfach das der Zelle #2
df_test     = cell_data_list[1]["test_data_orig"]

# Konvertierung zu Numpy
trainval_array = to_input_array(df_trainval)
test_array     = to_input_array(df_test)

print(f"Train/Val Gesamt: {len(df_trainval)} Zeilen (Zellen: {cell_names[0]} + {cell_names[1]})")
print(f"Test Set (nur {cell_names[1]}): {len(df_test)} Zeilen")

###############################################################################
# 5) Dataset-Definition
###############################################################################
seq_length = 60

class SequenceDataset(Dataset):
    """
    Gibt (seq, next_value) zurück, 
      X[t] = (Voltage[t], Current[t], SOC[t])  (über seq_length Timesteps)
      y[t] = SOC[t+1].
    """
    def __init__(self, data_array, seq_len=60):
        self.data_array = data_array
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data_array) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.data_array[idx : idx + self.seq_len]   # shape (seq_len, 3)
        y_val = self.data_array[idx + self.seq_len, 2]      # Spalte 2 = SOC
        x_seq_t = torch.tensor(x_seq, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        return x_seq_t, y_val_t

trainval_dataset = SequenceDataset(trainval_array, seq_len=seq_length)
test_dataset     = SequenceDataset(test_array,     seq_len=seq_length)

trainval_loader = DataLoader(trainval_dataset, batch_size=32, shuffle=True)
test_loader     = DataLoader(test_dataset,     batch_size=32, shuffle=False)


###############################################################################
# 6) Einfaches LSTM-Modell mittels pytorch_forecasting.models.nn.rnn.LSTM
###############################################################################
class LSTMSOCModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=1, batch_first=True):
        super().__init__()
        # ForecastingLSTM
        self.lstm = ForecastingLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        # Einfacher Linear-Layer für 1-D-Output (SOC)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x.shape = (batch_size, seq_length, 3)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Wir nehmen den letzten Zeitschritt:
        last_out = lstm_out[:, -1, :]    # (batch_size, hidden_size)
        soc_pred = self.fc(last_out)     # (batch_size, 1)
        return soc_pred.squeeze(-1)      # -> (batch_size,)

# Initialisiere Modell, Optimizer, Loss
model = LSTMSOCModel(input_size=3, hidden_size=32, num_layers=1, batch_first=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

###############################################################################
# 7) Training Loop
###############################################################################
epochs = 20
model.train()

for epoch in range(epochs):
    train_losses = []
    for x_batch, y_batch in trainval_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    print(f"Epoch [{epoch+1}/{epochs}] - Train MSE: {np.mean(train_losses):.6f}")

###############################################################################
# 8) Autoregressive Test-Vorhersage auf dem Test-Datensatz der 2. Zelle
###############################################################################
def predict_autoregressive(model, data_array, seq_len=60):
    """
    Nutzt die ersten seq_len SOC-Werte aus data_array als "Startfenster".
    Anschließend wird SOC[t+1] vom Modell vorhergesagt und zurückgeschrieben,
    so dass immer der neueste Wert als Input genutzt wird (autoregessiv).
    """
    model.eval()
    data_clone = data_array.copy()
    preds = np.full(len(data_clone), np.nan)

    with torch.no_grad():
        for i in range(seq_len, len(data_clone)):
            input_seq = data_clone[i-seq_len : i]  # (seq_len, 3)
            x_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
            pred_soc = model(x_t).item()
            preds[i] = pred_soc
            data_clone[i, 2] = pred_soc  # SOC in Spalte 2 überschreiben

    return preds

preds_test = predict_autoregressive(model, test_array, seq_len=seq_length)

###############################################################################
# 9) Plot: Ground Truth vs. Prediction im Testbereich der zweiten Zelle
###############################################################################
df_test_orig_2nd = cell_data_list[1]["test_data_orig"]  # Test-Daten der zweiten Zelle
time_test = df_test_orig_2nd['timestamp'].values
gt_test = test_array[:, 2]  # SOC-Spalte

plt.figure(figsize=(14, 5))
plt.plot(time_test, gt_test, label=f"Ground Truth SOC (Zelle {cell_names[1]})", color='k')
plt.plot(time_test, preds_test, label="Predicted SOC (autoregressive)", color='r', alpha=0.7)
plt.title(f"Autoregressive SOC-Vorhersage - Testblock für Zelle {cell_names[1]}")
plt.xlabel("Time")
plt.ylabel("SOC")
plt.legend()
plt.tight_layout()
plt.show()
