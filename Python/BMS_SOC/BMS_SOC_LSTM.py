import os
import sys
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
from tqdm import tqdm  # Neuer Import für den Fortschrittsbalken

# Neues Device-Setup (nach Imports)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
# 1) Laden der Daten
###############################################################################

def load_all_cell_data(data_dir: Path):
    """
    Durchsucht das Verzeichnis `data_dir` nach allen Unterordnern,
    die mit 'MGFarm_18650_C' beginnen. In jedem solchen Ordner wird
    die Datei 'df.parquet' eingelesen (falls vorhanden).

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
            # Beispiel: Ordnername "MGFarm_18650_C01" -> Zellname: "C01"
            cell_name = folder.name.replace("MGFarm_18650_", "")
            df_path = folder / 'df.parquet'
            if df_path.exists():
                df = pd.read_parquet(df_path)
                dataframes[cell_name] = df
                print(f"Loaded: {folder.name} -> Key: {cell_name}")
            else:
                print(f"Warning: No df.parquet found in {folder.name}")
    return dataframes

data_dir = Path('/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes')
cell_data = load_all_cell_data(data_dir)

if not cell_data:
    raise ValueError("Keine Zellen gefunden; bitte prüfen, ob Unterordner vorhanden sind.")

# Alle DataFrames zu einem großen kombinieren:
# z.B. df_full = pd.concat(list(cell_data.values()), ignore_index=True)
# Fürs Debuggen / Testen kann man auch nur einen Teil nehmen.
df_full = pd.concat(cell_data.values(), ignore_index=True)
print("Anzahl Zeilen (alle Zellen kombiniert):", len(df_full))

# Wir definieren wieder ein "Test-Fenster" (df_test) in der Mitte:
sample_size = int(len(df_full) * 0.01)  # 1% für Test (zum Beispiel)
mid_start = (len(df_full) - sample_size) // 2
df_test = df_full.iloc[mid_start : mid_start + sample_size].copy()

# Timestamp in Datetime umwandeln
df_test.loc[:, 'timestamp'] = pd.to_datetime(df_test['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
df_test_orig = df_test.copy()

print(f"Test-Datensatz-Größe (konstant): {len(df_test)}")


###############################################################################
# 2) Vorbereitung: Funktion, um aus df_test_orig jeweils 
#    ein (Train+Val)-Set via Zufall zu ziehen & zu skalieren
###############################################################################
def create_train_val_split_from_test(df_test_orig, frac=0.8, random_state=None):
    """
    Ziehe zufällig einen Teil aus df_test_orig, 
    splitte in 80% (train) und 20% (val),
    skaliere Voltage/Current.
    """
    # 1) Einen Teil ziehen (z.B. 80% von df_test_orig)
    n_samples = int(len(df_test_orig) * frac)
    df_train_val = df_test_orig.sample(n=n_samples, replace=False, random_state=random_state).copy()

    # 2) 80/20 in train/val
    total = len(df_train_val)
    cut = int(total * 0.8)

    # Optional sortieren nach Index, damit die Reihenfolge zeitlich konsistent ist
    df_train_val_sorted = df_train_val.sort_index()
    train_data_orig = df_train_val_sorted.iloc[:cut].copy()
    val_data_orig   = df_train_val_sorted.iloc[cut:].copy()

    # 3) Skalierung (nur Voltage & Current)
    scaler = StandardScaler()
    features = ['Voltage[V]', 'Current[A]']

    train_data = train_data_orig.copy()
    val_data   = val_data_orig.copy()

    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features]   = scaler.transform(val_data[features])

    return train_data, val_data, scaler


###############################################################################
# 3) Dataset-Klasse und Modell (LSTM)
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
        x_seq = self.data_array[idx : idx + self.seq_len]  # shape (seq_len, 3)
        y_val = self.data_array[idx + self.seq_len, 2]     # Spalte 2 = SOC
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
        # Ersetze for-loop durch tqdm-wrapped Loop
        for i in tqdm(range(seq_len, len(data_clone)), desc="Prediction"):
            input_seq = data_clone[i-seq_len : i]  # (seq_len, 3)
            x_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)  # shape (1, seq_len, 3)
            
            pred_soc = model(x_t).item()
            preds[i] = pred_soc
            # Autoregressiv: SOC-Spalte überschreiben
            data_clone[i, 2] = pred_soc

    return preds


###############################################################################
# 4) Hauptteil: Modell anlegen und iterativ trainieren
###############################################################################

# a) Initialisiere einmal das Modell und den Optimizer
model = LSTMSOCModel(input_size=3, hidden_size=32, num_layers=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

seq_length = 60
NUM_ITERATIONS = 100     # Wie oft wir das "iterative" Training wiederholen
EPOCHS_PER_ITER = 100    # Wie viele Epochen pro Iteration (klein für schnellen Test)

# (Optional) Ordner fürs Speichern
os.makedirs("models/LSTM", exist_ok=True)
model_save_path = "models/LSTM/lstm_soc_model_trained.pth"

for iteration in range(1, NUM_ITERATIONS+1):
    print(f"\n=== Iteration {iteration}/{NUM_ITERATIONS} ===")

    # 1) Erstelle Trainings- und Validierungs-Datensatz per Zufall aus df_test
    train_data, val_data, scaler = create_train_val_split_from_test(
        df_test_orig, 
        frac=0.8,            # z.B. 80% von df_test_orig als "Train+Val"
        random_state=None    # Keinen fixen Seed -> jedes Mal anderer Split
    )
    
    # 2) Kombiniere (Train + Val) zu einem DataFrame (fürs Training)
    df_trainval = pd.concat([train_data, val_data], axis=0).reset_index(drop=True)
    
    # 3) Erstelle Numpy-Arrays: [Voltage, Current, SOC]
    trainval_array = df_trainval[["Voltage[V]", "Current[A]", "SOC_ZHU"]].values
    
    # 4) Test-Daten für diese Iteration: 
    #    Voltage/Current skalieren mit dem frisch gelernten Scaler
    df_test_scaled = df_test.copy()
    df_test_scaled[['Voltage[V]', 'Current[A]']] = scaler.transform(
        df_test_scaled[['Voltage[V]', 'Current[A]']]
    )
    test_array = df_test_scaled[["Voltage[V]", "Current[A]", "SOC_ZHU"]].values
    
    # 5) DataLoader für Training & Test
    trainval_dataset = SequenceDataset(trainval_array, seq_len=seq_length)
    trainval_loader = DataLoader(trainval_dataset, batch_size=32, shuffle=True)
    
    test_dataset = SequenceDataset(test_array, seq_len=seq_length)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 6) Training auf diesem (zufällig gezogenen) Datensatz
    model.train()
    for epoch in range(1, EPOCHS_PER_ITER+1):
        train_losses = []
        # Ersetze den Batch-Loop durch tqdm-wrapped Loop
        for x_batch, y_batch in tqdm(trainval_loader, desc=f"Epoch {epoch}/{EPOCHS_PER_ITER}"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print(f"  Epoch [{epoch}/{EPOCHS_PER_ITER}] - Train MSE: {np.mean(train_losses):.4f}")
    
    # 7) Zwischendurch speichern
    torch.save(model.state_dict(), model_save_path)
    print(f"Modell gespeichert: {model_save_path}")
    
    # 8) Testen + Plotten
    preds_test = predict_autoregressive(model, test_array, seq_len=seq_length)
    gt_test = df_test_orig["SOC_ZHU"].values  # Ground Truth SOC unskaliert
    time_test = df_test_orig['timestamp'].values
    
    plt.figure(figsize=(14, 5))
    plt.plot(time_test, gt_test, label="Ground Truth SOC", color='k')
    plt.plot(time_test, preds_test, label="Predicted SOC (autoregressive)", color='r', alpha=0.7)
    plt.title(f"Autoregressive SOC-Vorhersage - Iteration {iteration} (alle Zellen zusammen)")
    plt.xlabel("Time")
    plt.ylabel("SOC")
    plt.legend()
    plt.tight_layout()

    # Plot speichern
    save_name = f"models/LSTM/model_iterativ_{iteration}.png"
    plt.savefig(save_name)
    print(f"Plot gespeichert unter: {save_name}")
    plt.show()
    # plt.close()  # Falls du kein Popup möchtest, stattdessen plt.close()

print("Iteratives Training abgeschlossen.")
