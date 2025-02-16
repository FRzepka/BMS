import os
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Aus pytorch_forecasting
from pytorch_forecasting.models.nn.rnn import LSTM as ForecastingLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
# 1) Daten laden (df_scaled)
###############################################################################
def load_cell_data(data_dir: Path):
    """Lade df_scaled.parquet aus dem Ordner 'MGFarm_18650_C01'."""
    dataframes = {}
    folder = data_dir / "MGFarm_18650_C01"
    if folder.exists() and folder.is_dir():
        df_path = folder / 'df_scaled.parquet'
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

cell_keys = sorted(cell_data.keys())[:1]
if len(cell_keys) < 1:
    raise ValueError("Keine Zelle gefunden; bitte prüfen.")

cell_name = cell_keys[0]  # z.B. 'C01'
df_full = cell_data[cell_name]

# Wir nehmen 0,1% der Daten
sample_size = int(len(df_full) * 0.01)
print(f"Verwende 0.1% der Daten: {sample_size} Zeilen (von {len(df_full)})")

# Schneide den vorderen Teil raus (head), damit wir "klassisch" den vorderen Bereich nehmen
df_small = df_full.head(sample_size).copy()

# 10% davon für Test (hinterer Teil)
test_size = int(len(df_small) * 0.1)
trainval_size = len(df_small) - test_size

df_trainval = df_small.head(trainval_size)
df_test     = df_small.tail(test_size)

# Zeit in datetime konvertieren (nur für Plot und Übersicht)
df_trainval.loc[:, 'timestamp'] = pd.to_datetime(df_trainval['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
df_test.loc[:, 'timestamp']     = pd.to_datetime(df_test['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])

# Innerhalb der trainval-Daten 80% -> Train, 20% -> Val (zeitbasiert)
tv_cut = int(trainval_size * 0.8)
train_data_orig = df_trainval.iloc[:tv_cut]
val_data_orig   = df_trainval.iloc[tv_cut:]

print("\n--- Datengrößen ---")
print(f"Gesamt: 0,1% = {sample_size} Zeilen")
print(f"Train/Val gesamt: {len(df_trainval)}")
print(f"  -> Training:   {len(train_data_orig)}")
print(f"  -> Validation: {len(val_data_orig)}")
print(f"Test:           {len(df_test)}")
print("---------------")

###############################################################################
# 2) Plot: Train / Val / Test (vor Training)
###############################################################################
plt.figure(figsize=(16,10))

# Beispiel: plotte Voltage, SOC, Current jeweils in einem Subplot
# Du kannst natürlich stattdessen direkt die Spalten "Scaled_Voltage[V]" etc. nehmen.
# Beachte: Plot-Methoden sind schematisch, pass ggf. an.
plt.subplot(3,1,1)
plt.plot(train_data_orig['timestamp'], train_data_orig['Scaled_Voltage[V]'], 'b-', label='Train')
plt.plot(val_data_orig['timestamp'],   val_data_orig['Scaled_Voltage[V]'], 'orange', label='Val')
plt.plot(df_test['timestamp'],         df_test['Scaled_Voltage[V]'], 'g-', label='Test')
plt.title(f"Voltage (Scaled) - 0.1% Daten, Zelle {cell_name}")
plt.legend()

plt.subplot(3,1,2)
plt.plot(train_data_orig['timestamp'], train_data_orig['SOC_ZHU'], 'b-', label='Train')
plt.plot(val_data_orig['timestamp'],   val_data_orig['SOC_ZHU'], 'orange', label='Val')
plt.plot(df_test['timestamp'],         df_test['SOC_ZHU'], 'g-', label='Test')
plt.title("SOC_ZHU")
plt.legend()

plt.subplot(3,1,3)
plt.plot(train_data_orig['timestamp'], train_data_orig['Scaled_Current[A]'], 'b-', label='Train')
plt.plot(val_data_orig['timestamp'],   val_data_orig['Scaled_Current[A]'], 'orange', label='Val')
plt.plot(df_test['timestamp'],         df_test['Scaled_Current[A]'], 'g-', label='Test')
plt.title("Current (Scaled)")
plt.legend()

plt.tight_layout()
plt.show()

###############################################################################
# 3) Dataset-Klassen
###############################################################################
seq_length = 120

class SequenceDataset(Dataset):
    """
    Baut (seq, label) Paare:
    - seq = 60 aufeinanderfolgende Zeitschritte (Voltage, Current, SOC)
    - label = SOC am Schritt (idx + seq_length)
    
    seq_length = 60 bedeutet also 60-Sekunden-Päckchen, 
    falls deine Daten 1 Sekunde Takt haben.
    """
    def __init__(self, df, seq_len=60):
        # Extrahiere die nötigen Spalten
        data_array = df[["Scaled_Voltage[V]", "Scaled_Current[A]", "SOC_ZHU"]].values
        self.data_array = data_array
        self.seq_len = seq_len

    def __len__(self):
        # Beispielsweise: Wenn 1000 Zeilen, hat man 1000 - seq_len Trainingssamples
        return len(self.data_array) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.data_array[idx : idx + self.seq_len]        # shape (seq_len, 3)
        y_val = self.data_array[idx + self.seq_len, 2]           # SOC_ZHU liegt in Spalte 2
        x_seq_t = torch.tensor(x_seq, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        return x_seq_t, y_val_t

# Erstelle unsere Datasets
train_dataset = SequenceDataset(train_data_orig, seq_len=seq_length)
val_dataset   = SequenceDataset(val_data_orig,   seq_len=seq_length)
test_dataset  = SequenceDataset(df_test,         seq_len=seq_length)

# DataLoader: shuffle=True für Train (mischt nur die Reihenfolge der Päckchen)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, drop_last=True)

###############################################################################
# 4) Einfaches LSTM-Modell
###############################################################################
class LSTMSOCModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=1, num_layers=1, batch_first=True):
        super().__init__()
        # ForecastingLSTM aus pytorch_forecasting
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
        # Letzten Zeitschritt rauspicken
        last_out = lstm_out[:, -1, :]    # (batch_size, hidden_size)
        soc_pred = self.fc(last_out)     # (batch_size, 1)
        return soc_pred.squeeze(-1)      # -> (batch_size,)

model = LSTMSOCModel(
    input_size=3,
    hidden_size=128,
    num_layers=1,
    batch_first=True
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0013789252608308762,
    weight_decay=6.578280304241413e-06
)
criterion = nn.MSELoss()

###############################################################################
# 5) Training Loop (mit Validation)
###############################################################################
epochs = 9

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
            val_loss = criterion(y_pred_val, y_val)
            val_losses.append(val_loss.item())
    mean_val_loss = np.mean(val_losses)

    print(f"Epoch [{epoch}/{epochs}] - "
          f"Train MSE: {mean_train_loss:.6f}, "
          f"Val MSE: {mean_val_loss:.6f}")

###############################################################################
# 6) Testphase: Autoregressive Vorhersage
###############################################################################
def predict_autoregressive(model, df, seq_len=10):
    """
    Wir nutzen einen "Sliding Window"-Ansatz:
    - Start mit den ersten seq_len Schritten aus df
    - Modell sagt SOC[t+1] voraus
    - Überschreiben in data_clone[i, 2]
    """
    model.eval()
    data_array = df[["Scaled_Voltage[V]", "Scaled_Current[A]", "SOC_ZHU"]].values
    data_clone = data_array.copy()
    preds = np.full(len(data_clone), np.nan)

    with torch.no_grad():
        for i in range(seq_len, len(data_clone)):
            input_seq = data_clone[i - seq_len : i]  # (seq_len, 3)
            x_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred_soc = model(x_t).item()
            preds[i] = pred_soc
            # Überschreibe die SOC-Spalte (index 2) mit dem Vorhersagewert
            data_clone[i, 2] = pred_soc

    return preds

preds_test = predict_autoregressive(model, df_test, seq_len=seq_length)

# Plot Test-Vorhersage
gt_test = df_test['SOC_ZHU'].values
t_test  = df_test['timestamp'].values

plt.figure(figsize=(14,5))
plt.plot(t_test, gt_test, label="Ground Truth SOC", color='k')
plt.plot(t_test, preds_test, label="Predicted SOC", color='r', alpha=0.7)
plt.title(f"Autoregressive SOC-Vorhersage - Test (Zelle: {cell_name})")
plt.xlabel("Time")
plt.ylabel("SOC (ZHU)")
plt.legend()
plt.tight_layout()

# Ggf. speichern
import os
from pathlib import Path

# Hier anpassen: Pfad, wo du Modell & Plot sichern willst
current_dir = Path(__file__).parent
models_dir = current_dir / "models"
os.makedirs(models_dir, exist_ok=True)
plot_file = models_dir / "prediction_test.png"
plt.savefig(plot_file)
print(f"Plot gespeichert unter: {plot_file}")
plt.show()

# Modell speichern
model_path = models_dir / "lstm_soc_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Modell gespeichert unter: {model_path}")
