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
from sklearn.model_selection import KFold
from tqdm import tqdm  # Importiere tqdm

###############################################################################
# 1) Laden der Daten
###############################################################################

def load_cell_data(data_dir: Path):
    """Lade nur die df.parquet aus dem Unterordner 'MGFarm_18650_C01'.
       Der Schlüssel im Rückgabedict ist der Zellname (z.B. 'C01')."""
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

data_dir = Path('/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes')
cell_data = load_cell_data(data_dir)

# Verwende nur die erste Zelle (C01)
cell_keys = sorted(cell_data.keys())[:1]
if len(cell_keys) < 1:
    raise ValueError("Keine Zelle gefunden; bitte prüfen.")

train_cell = cell_keys[0]  # z.B. 'C01'
df_full = cell_data[train_cell]

###############################################################################
# 2) Vorbereitung der Datenaufteilung in Trainings-, Validierungs- und Testmengen
###############################################################################

def create_splits(df, split_size=0.001, num_train_splits=10, random_state=42):
    """Erstellt zufällige Splits der Daten für Training, Validierung und Test."""
    
    # 1) Erstelle Liste von zufälligen Indizes für die Splits
    num_splits = len(df) // int(len(df) * split_size)  # Anzahl möglicher Splits
    all_indices = np.arange(len(df))
    np.random.seed(random_state)
    np.random.shuffle(all_indices)
    
    split_indices = []
    for i in range(num_splits):
        start = int(i * len(df) * split_size)
        end   = int((i+1) * len(df) * split_size)
        split_indices.append(all_indices[start:end])
    
    # 2) Wähle zufällig Trainings-Splits aus
    train_indices = np.random.choice(len(split_indices), size=num_train_splits, replace=False)
    train_splits = [df.iloc[split_indices[i]].copy() for i in train_indices]
    
    # 3) Erstelle Test-Split (alle übrigen)
    test_indices = np.array([i for i in range(len(split_indices)) if i not in train_indices])
    
    # Wenn keine Testdaten vorhanden sind, erstelle einen leeren DataFrame
    if len(test_indices) > 0:
        # Wähle den ersten Index aus den Testindizes aus
        test_index = test_indices[0]
        # Erstelle den Test-Split mit dem ausgewählten Index
        test_split = df.iloc[split_indices[test_index]].copy()
    else:
        # Erstelle einen leeren DataFrame, wenn keine Testdaten vorhanden sind
        test_split = pd.DataFrame()
    
    return train_splits, test_split

# Erstelle die Splits
train_splits, test_split = create_splits(df_full, split_size=0.001, num_train_splits=10, random_state=42)

print(f"Anzahl Trainings-Splits: {len(train_splits)}")
print(f"Größe eines Trainings-Splits: {len(train_splits[0])}")
print(f"Größe des Test-Splits: {len(test_split)}")

###############################################################################
# 3) Datensatz-Klasse und Modell
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
        for i in range(seq_len, len(data_clone)):
            input_seq = data_clone[i-seq_len : i]  # (seq_len, 3)
            x_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # shape (1, seq_len, 3)
            
            pred_soc = model(x_t).item()
            preds[i] = pred_soc
            # Autoregressiv: SOC-Spalte überschreiben
            data_clone[i, 2] = pred_soc

    return preds

###############################################################################
# 4) Hauptteil: Modell anlegen und trainieren
###############################################################################

# a) Initialisierung
model = LSTMSOCModel(input_size=3, hidden_size=32, num_layers=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
seq_length = 60
EPOCHS = 20

# b) K-Fold Cross-Validation
kf = KFold(n_splits=len(train_splits), shuffle=False)
for fold, (train_index, val_index) in enumerate(tqdm(kf.split(train_splits), total=len(train_splits), desc="Folds"), start=1):
    print(f"\n=== Fold {fold}/{len(train_splits)} ===")
    
    # 1) Trainings- und Validierungsdaten vorbereiten
    # Da kf.split() Indizes liefert, die sich auf die Liste train_splits beziehen,
    # müssen wir die entsprechenden DataFrames auswählen.
    train_data_list = [train_splits[i] for i in train_index]
    val_data   = train_splits[val_index[0]]  # val_index ist ein Array, wir nehmen das erste Element
    
    # 2) Skalierung (Voltage & Current).  Wichtig: Fit nur auf den Trainingsdaten!
    scaler = StandardScaler()
    features = ['Voltage[V]', 'Current[A]']
    
    # Kombiniere alle Trainingsdaten für das Fitten des Scalers
    train_data_combined = pd.concat(train_data_list, axis=0)
    train_data_combined[features] = scaler.fit_transform(train_data_combined[features])
    
    # Transformiere die einzelnen Trainings-DataFrames
    for i in range(len(train_data_list)):
        train_data_list[i][features] = scaler.transform(train_data_list[i][features])
    
    # Transformiere die Validierungsdaten
    val_data[features] = scaler.transform(val_data[features])
    
    # 3) Erstelle Numpy-Arrays: [Voltage, Current, SOC]
    train_arrays = [df[["Voltage[V]", "Current[A]", "SOC_ZHU"]].values for df in train_data_list]
    val_array   = val_data[["Voltage[V]", "Current[A]", "SOC_ZHU"]].values
    
    # 4) Erzeuge PyTorch-Datasets und DataLoader
    # Kombiniere die einzelnen Trainings-Arrays zu einem großen Array
    train_array_combined = np.concatenate(train_arrays, axis=0)
    train_dataset = SequenceDataset(train_array_combined, seq_len=seq_length)
    val_dataset   = SequenceDataset(val_array,   seq_len=seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    
    # 5) Training
    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs", leave=True):
        model.train()
        train_losses = []
        for batch_idx, (x_batch, y_batch) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Batches", leave=False):
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        print(f"  Fold {fold} - Epoch [{epoch}/{EPOCHS}] - Train MSE: {np.mean(train_losses):.4f}")

# c) Testen
print("\n=== Test ===")

# 1) Skaliere den Test-Split mit dem gleichen Scaler, der beim letzten Training verwendet wurde
test_data = test_split.copy()
test_data[features] = scaler.transform(test_data[features])
test_array = test_data[["Voltage[V]", "Current[A]", "SOC_ZHU"]].values

# 2) Erstelle Test-Dataset und DataLoader
test_dataset = SequenceDataset(test_array, seq_len=seq_length)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# 3) Vorhersage auf dem Test-Datensatz
preds_test = predict_autoregressive(model, test_array, seq_len=seq_length)
gt_test = test_split["SOC_ZHU"].values  # Ground Truth SOC unskaliert
if 'timestamp' in test_split.columns:
    time_test = test_split['timestamp'].values
else:
    time_test = np.arange(len(gt_test))  # Verwende Dummy-Zeitreihe, falls keine Zeitstempel vorhanden

# d) Plotten der Ergebnisse
plt.figure(figsize=(14, 5))
plt.plot(time_test, gt_test, label="Ground Truth SOC", color='k')
plt.plot(time_test, preds_test, label="Predicted SOC (autoregressive)", color='r', alpha=0.7)
plt.title("Autoregressive SOC-Vorhersage - Test")
plt.xlabel("Time")
plt.ylabel("SOC")
plt.legend()
plt.tight_layout()

# Plot speichern
save_name = "model_test_one_run.png"
plt.savefig(save_name)
print(f"Plot gespeichert unter: {save_name}")
plt.show()
# plt.close()  # Falls du keinen plt.show() willst, dann statt show() -> close()

print("Training und Test abgeschlossen.")
