################################################################################
# Ein simples Beispiel-Skript, das
# 1) Genau die im vorgegebenen Code gezeigte Datenaufteilung verwendet.
# 2) Nun aber statt df.parquet -> df_scaled.parquet lädt.
# 3) Keine Skalierung mit StandardScaler mehr durchführt, da in df_scaled.parquet
#    bereits skaliert vorliegende Spalten genutzt werden.
################################################################################

import os
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Falls noch nicht installiert:
# pip install pytorch-forecasting
from pytorch_forecasting.models.nn.rnn import LSTM as ForecastingLSTM

# GPU Setup: Prüfe, ob CUDA verfügbar ist und gebe entsprechendes Feedback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
# 1) Daten laden und Aufteilen wie im vorgegebenen Code
#    ABER: Statt df.parquet -> df_scaled.parquet
###############################################################################

def load_cell_data(data_dir: Path):
    """Lade nur die df_scaled.parquet aus dem Unterordner 'MGFarm_18650_C01'.
       Der Schlüssel im Rückgabedict ist der Zellname (z.B. 'C01')."""
    dataframes = {}
    folder = data_dir / "MGFarm_18650_C01"
    if folder.exists() and folder.is_dir():
        df_path = folder / 'df_scaled.parquet'  # << Hier auf df_scaled.parquet geändert
        if df_path.exists():
            df = pd.read_parquet(df_path)
            dataframes["C01"] = df
            print(f"Loaded {folder.name} (df_scaled.parquet)")
        else:
            print(f"Warning: No df_scaled.parquet found in {folder.name}")
    else:
        print("Warning: Folder MGFarm_18650_C01 not found")
    return dataframes

# Pfad anpassen
data_dir = Path('/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes')
cell_data = load_cell_data(data_dir)

# Nur die erste Zelle verwenden
cell_keys = sorted(cell_data.keys())[:1]
if len(cell_keys) < 1:
    raise ValueError("Es wurde keine Zelle gefunden; bitte prüfen.")

train_cell = cell_keys[0]  # z.B. 'C01'
df_full = cell_data[train_cell]

# 0.1% der Daten als Train/Val, ein mittlerer Block als Test
sample_size = int(len(df_full) * 0.1)
df_train_val = df_full.head(sample_size)
mid_start = (len(df_full) - sample_size) // 2
df_test = df_full.iloc[mid_start : mid_start + sample_size]

# Timestamps konvertieren
df_train_val.loc[:, 'timestamp'] = pd.to_datetime(df_train_val['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
df_test.loc[:, 'timestamp'] = pd.to_datetime(df_test['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])

# Kopien anlegen, bevor wir sie verändern
df_train_val_orig = df_train_val.copy()
df_test_orig = df_test.copy()

# Train/Val Split (80%/20%)
total = len(df_train_val)
train_cut = int(total * 0.8)
train_data_orig = df_train_val_orig.iloc[:train_cut]
val_data_orig   = df_train_val_orig.iloc[train_cut:]

print(f"Total train/val sample size: {total}")
print(f"Training set size: {len(train_data_orig)}")
print(f"Validation set size: {len(val_data_orig)}")
print(f"Test set size: {len(df_test)}")

###############################################################################
# 2) Optional: Kleiner Plot zur Kontrolle (Original-Dataframes)
###############################################################################
plt.figure(figsize=(15,12))

# Bitte anpassen, falls du lieber die skalierten Spalten plotten möchtest.
# Hier plotten wir weiterhin "Scaled_Voltage[V]" etc., sofern gewünscht.
# Ansonsten kann man auch die Original-Spalten im df_scaled plotten, 
# je nachdem, wie du deine df_scaled.parquet organisiert hast.
plt.subplot(3,1,1)
plt.plot(train_data_orig['timestamp'], train_data_orig['Scaled_Voltage[V]'], 'b-', label='Training')
plt.plot(val_data_orig['timestamp'],   val_data_orig['Scaled_Voltage[V]'], 'orange', label='Validation')
plt.title('Scaled Voltage - Train/Val')
plt.xlabel('Time')
plt.ylabel('Scaled Voltage')
plt.legend()

plt.subplot(3,1,2)
plt.plot(train_data_orig['timestamp'], train_data_orig['SOC_ZHU'], 'b-', label='Training')
plt.plot(val_data_orig['timestamp'],   val_data_orig['SOC_ZHU'], 'orange', label='Validation')
plt.title('Scaled SOC - Train/Val')
plt.xlabel('Time')
plt.ylabel('Scaled SOC')
plt.legend()

plt.subplot(3,1,3)
plt.plot(train_data_orig['timestamp'], train_data_orig['Scaled_Current[A]'], 'b-', label='Training')
plt.plot(val_data_orig['timestamp'],   val_data_orig['Scaled_Current[A]'], 'orange', label='Validation')
plt.title('Scaled Current - Train/Val')
plt.xlabel('Time')
plt.ylabel('Scaled Current')
plt.legend()
plt.tight_layout()
plt.show()


###############################################################################
# 3) Datensätze (Dataset) für (Scaled_Voltage, Scaled_Current, Scaled_SOC) Sequenzen
###############################################################################
seq_length = 60

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
        x_seq = self.data_array[idx : idx + self.seq_len]   # shape (seq_len, 3)
        y_val = self.data_array[idx + self.seq_len, 2]      # Spalte 2 = SOC
        x_seq_t = torch.tensor(x_seq, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        return x_seq_t, y_val_t

# Wir nehmen jetzt direkt die bereits skalierten Spalten:
# ['Scaled_Voltage[V]', 'Scaled_Current[A]', 'Scaled_SOC_ZHU']
train_data = train_data_orig.reset_index(drop=True)
val_data   = val_data_orig.reset_index(drop=True)
test_data  = df_test.reset_index(drop=True)

# Erstelle aus train+val einen gemeinsamen "TrainVal"-Satz
df_trainval = pd.concat([train_data, val_data], axis=0).reset_index(drop=True)

# ACHTUNG: Passe das an, wenn deine Spaltennamen abweichen!
trainval_array = df_trainval[["Scaled_Voltage[V]", "Scaled_Current[A]", "SOC_ZHU"]].values
test_array     = test_data[["Scaled_Voltage[V]", "Scaled_Current[A]", "SOC_ZHU"]].values

trainval_dataset = SequenceDataset(trainval_array, seq_len=seq_length)
test_dataset     = SequenceDataset(test_array,     seq_len=seq_length)

trainval_loader = DataLoader(trainval_dataset, batch_size=32, shuffle=True)
test_loader     = DataLoader(test_dataset,     batch_size=32, shuffle=False)


###############################################################################
# 4) Einfaches Modell mit pytorch_forecasting.models.nn.rnn.LSTM
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
        # Ein einfacher Linear-Layer für 1-D-Output (SOC)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x.shape = (batch_size, seq_length, 3)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Wir nehmen den letzten Zeitschritt:
        last_out = lstm_out[:, -1, :]    # (batch_size, hidden_size)
        soc_pred = self.fc(last_out)     # (batch_size, 1)
        return soc_pred.squeeze(-1)      # -> (batch_size,)

# Modell, Optimizer, Loss; Modell auf GPU
model = LSTMSOCModel(input_size=3, hidden_size=32, num_layers=1, batch_first=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

###############################################################################
# 5) Einfacher Trainingsloop
###############################################################################
epochs = 20
model.train()

for epoch in range(epochs):
    train_losses = []
    for x_batch, y_batch in trainval_loader:
        optimizer.zero_grad()
        # Sende Batch auf GPU:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    print(f"Epoch [{epoch+1}/{epochs}] - Train MSE: {np.mean(train_losses):.6f}")


###############################################################################
# 6) Autoregressive Vorhersage auf dem Test-Datensatz
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
            # Sende Eingabe an GPU:
            x_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred_soc = model(x_t).item()
            preds[i] = pred_soc
            data_clone[i, 2] = pred_soc  # SOC in Spalte 2 überschreiben

    return preds

preds_test = predict_autoregressive(model, test_array, seq_len=seq_length)

###############################################################################
# 7) Plot: Ground Truth vs. Prediction im Testbereich
###############################################################################
gt_test = test_array[:, 2]
time_test = df_test_orig['timestamp'].values

plt.figure(figsize=(14, 5))
plt.plot(time_test, gt_test, label="Ground Truth SOC", color='k')
plt.plot(time_test, preds_test, label="Predicted SOC (autoregressive)", color='r', alpha=0.7)
plt.title(f"Autoregressive SOC-Vorhersage - Testblock (Zelle: {train_cell})")
plt.xlabel("Time")
plt.ylabel("SOC")
plt.legend()
plt.tight_layout()

# Speichere Diagramm im "/models"-Ordner
from pathlib import Path
import os

current_dir = Path(__file__).parent
models_dir = current_dir / "models"
os.makedirs(models_dir, exist_ok=True)
plot_file = models_dir / "prediction.png"
plt.savefig(plot_file)
print(f"Plot gespeichert unter: {plot_file}")
plt.show()

# Speichere das Modell im gleichen "/models"-Ordner
import torch
model_path = models_dir / "lstm_soc_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Modell gespeichert unter: {model_path}")
