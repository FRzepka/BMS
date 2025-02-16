import os
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Aus pytorch_forecasting
from pytorch_forecasting.models.nn.rnn import LSTM as ForecastingLSTM

###############################################################################
# 1) Laden der Daten, Reduktion auf 10%, zeitbasiertes Split in [Train, Val, Test]
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
            print(f"Loaded {folder.name}")
        else:
            print(f"Warning: No df.parquet found in {folder.name}")
    else:
        print("Warning: Folder MGFarm_18650_C01 not found")
    return dataframes

# Anpassen an deinen Speicherort:
data_dir = Path('/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes')
cell_data = load_cell_data(data_dir)

# Wir gehen davon aus, dass wir mindestens eine Zelle haben
cell_keys = sorted(cell_data.keys())[:1]
if len(cell_keys) < 1:
    raise ValueError("Keine Zelle gefunden; bitte prüfen.")

cell_name = cell_keys[0]  # z.B. 'C01'
df_full = cell_data[cell_name]

# 10% der Daten als "df_small"
sample_size = int(len(df_full) * 0.2)
df_small = df_full.head(sample_size).copy()  # vorderer Teil

print(f"Gesamtdaten: {len(df_full)}, wir nehmen 10% = {sample_size} Zeilen.")

# Zeit in datetime konvertieren (nur für Plot oder Zeit-spezifische Analysen)
df_small.loc[:, 'timestamp'] = pd.to_datetime(df_small['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])

# Zeitbasiertes Split: Train(80%), Val(10%), Test(10%) – in dieser Reihenfolge
len_small = len(df_small)
train_end = int(len_small * 0.8)  # bis hier Train
val_end   = int(len_small * 0.9)  # bis hier Val, danach Test

df_train = df_small.iloc[:train_end]
df_val   = df_small.iloc[train_end:val_end]
df_test  = df_small.iloc[val_end:]

print(f"Train: {len(df_train)}  |  Val: {len(df_val)}  |  Test: {len(df_test)}")

###############################################################################
# 2) Skalierung von Voltage & Current (NICHT SOC, da 0..1)
###############################################################################
scaler = StandardScaler()
features_to_scale = ['Voltage[V]', 'Current[A]']

# Fit nur auf Training
scaler.fit(df_train[features_to_scale])

# Transformation
df_train_scaled = df_train.copy()
df_val_scaled   = df_val.copy()
df_test_scaled  = df_test.copy()

df_train_scaled[features_to_scale] = scaler.transform(df_train_scaled[features_to_scale])
df_val_scaled[features_to_scale]   = scaler.transform(df_val_scaled[features_to_scale])
df_test_scaled[features_to_scale]  = scaler.transform(df_test_scaled[features_to_scale])

###############################################################################
# 3) Dataset-Klasse (autoregessives Fenster)
###############################################################################
class SequenceDataset(Dataset):
    """
    (X[t], y[t]) mit seq_len Zeitschritten.
    - X[t] = [Voltage, Current, SOC] für t..t+seq_len-1
    - y[t] = SOC an (t + seq_len)
    """
    def __init__(self, df, seq_len=60):
        # Hier NICHT SOC skalieren, da er bereits in 0..1 liegt
        self.seq_len = seq_len
        data_array = df[["Voltage[V]", "Current[A]", "SOC_ZHU"]].values
        self.data = data_array

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.data[idx : idx + self.seq_len]     # shape (seq_len, 3)
        y_val = self.data[idx + self.seq_len, 2]        # SOC in Spalte 2
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

# Erstellen der Datasets
seq_length = 60
train_dataset = SequenceDataset(df_train_scaled, seq_len=seq_length)
val_dataset   = SequenceDataset(df_val_scaled,   seq_len=seq_length)
test_dataset  = SequenceDataset(df_test_scaled,  seq_len=seq_length)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, drop_last=True)

###############################################################################
# 4) LSTM-Modell (pytorch_forecasting)
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
        # x: (batch_size, seq_len, 3)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        soc_pred = self.fc(last_out)
        return soc_pred.squeeze(-1)    # (batch_size,)

model = LSTMSOCModel(input_size=3, hidden_size=32, num_layers=1, batch_first=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

###############################################################################
# 5) Training mit Validation + Early Stopping
###############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 5
patience = 3  # Early Stopping Geduld

best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

for epoch in range(1, epochs+1):
    # --- TRAIN ---
    model.train()
    train_losses = []
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    mean_train_loss = np.mean(train_losses)

    # --- VALIDATION ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            y_pred_val = model(x_val)
            v_loss = criterion(y_pred_val, y_val)
            val_losses.append(v_loss.item())
    mean_val_loss = np.mean(val_losses)

    # Early Stopping Check
    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        best_model_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} because val_loss not improved.")
            break

    print(f"Epoch {epoch:03d}/{epochs}, "
          f"Train MSE: {mean_train_loss:.6f}, "
          f"Val MSE: {mean_val_loss:.6f}, "
          f"NoImprove: {epochs_no_improve}")

# Wiederherstellen des besten Modells
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\nBest model reloaded with val_loss = {best_val_loss:.6f}")

###############################################################################
# 6) Autoregressive Vorhersage (ohne Ground-Truth im Test)
###############################################################################
def predict_autoregressive(model, df, seq_len=60):
    """
    Autoregressiv: Es wird NICHT der echte SOC zur nächsten Schrittsvorhersage genutzt,
    sondern immer das zuletzt vorhergesagte SOC. 

    Dazu:
    - Wir kopieren die Daten in data_clone
    - Start: seq_len echte Werte
    - Ab i=seq_len nutzen wir nur noch die Vorhersagen im SOC-Feld (Spalte 2)
    """
    model.eval()
    data_array = df[["Voltage[V]", "Current[A]", "SOC_ZHU"]].values.copy()
    preds = np.full(len(data_array), np.nan)

    with torch.no_grad():
        for i in range(seq_len, len(data_array)):
            input_seq = data_array[i - seq_len : i]  # shape (seq_len, 3)
            x_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred_soc = model(x_t).item()
            preds[i] = pred_soc
            # Schreibe den SOC ins "SOC-ZHU"-Feld (Spalte 2) zurück,
            # damit es beim nächsten Schritt als Input dient
            data_array[i, 2] = pred_soc

    return preds

# Autoregressive Vorhersage auf Testset (bereits skaliert in df_test_scaled)
preds_test = predict_autoregressive(model, df_test_scaled, seq_len=seq_length)

###############################################################################
# 7) Plot: Ground Truth vs. Prediction (Test-Bereich)
###############################################################################
gt_test = df_test['SOC_ZHU'].values  # "echter" SOC (unskalierter, da 0..1)
t_test  = df_test['timestamp'].values

plt.figure(figsize=(12,5))
plt.plot(t_test, gt_test, label="Ground Truth SOC", color='k')
plt.plot(t_test, preds_test, label="Predicted SOC (autoregressive)", color='r', alpha=0.7)
plt.title(f"Autoregressive SOC-Vorhersage - Test (Zelle: {cell_name})")
plt.xlabel("Time")
plt.ylabel("SOC (ZHU)")
plt.legend()
plt.tight_layout()

# Modell + Plot speichern (optional)
from pathlib import Path
models_dir = Path(__file__).parent / "models"
os.makedirs(models_dir, exist_ok=True)

plot_file = models_dir / "prediction_test.png"
plt.savefig(plot_file)
plt.show()
print(f"Test-Plot gespeichert unter: {plot_file}")

model_path = models_dir / "best_lstm_soc_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Bestes Modell gespeichert unter: {model_path}")
