import os
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pytorch_forecasting.models.nn.rnn import LSTM as ForecastingLSTM
from sklearn.model_selection import TimeSeriesSplit

import optuna
from tqdm import tqdm

###############################################################################
# 1) Daten laden (df_scaled) -- unverändert aus deinem Code
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

df_small = df_full.head(sample_size).copy()

# 10% davon für Test (hinterer Teil)
test_size = int(len(df_small) * 0.1)
trainval_size = len(df_small) - test_size

df_trainval = df_small.head(trainval_size)
df_test     = df_small.tail(test_size)

# Zeit in datetime konvertieren (nur für Plot und Übersicht)
df_trainval.loc[:, 'timestamp'] = pd.to_datetime(df_trainval['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
df_test.loc[:, 'timestamp']     = pd.to_datetime(df_test['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])

print("\n--- Datengrößen ---")
print(f"Gesamt: 0,1% = {sample_size} Zeilen")
print(f"Train/Val gesamt: {len(df_trainval)}")
print(f"Test:             {len(df_test)}")
print("---------------")

###############################################################################
# Plot: Vor dem Training
###############################################################################
plt.figure(figsize=(16,10))

plt.subplot(3,1,1)
plt.plot(df_trainval['timestamp'], df_trainval['Scaled_Voltage[V]'], 'b-', label='Train/Val')
plt.plot(df_test['timestamp'],     df_test['Scaled_Voltage[V]'], 'g-', label='Test')
plt.title(f"Voltage (Scaled) - 0.1% Daten, Zelle {cell_name}")
plt.legend()

plt.subplot(3,1,2)
plt.plot(df_trainval['timestamp'], df_trainval['SOC_ZHU'], 'b-', label='Train/Val')
plt.plot(df_test['timestamp'],     df_test['SOC_ZHU'], 'g-', label='Test')
plt.title("SOC_ZHU")
plt.legend()

plt.subplot(3,1,3)
plt.plot(df_trainval['timestamp'], df_trainval['Scaled_Current[A]'], 'b-', label='Train/Val')
plt.plot(df_test['timestamp'],     df_test['Scaled_Current[A]'], 'g-', label='Test')
plt.title("Current (Scaled)")
plt.legend()

plt.tight_layout()
plt.show()

###############################################################################
# 2) SequenceDataset-Klasse
###############################################################################
class SequenceDataset(Dataset):
    """
    Baut (seq, label) Paare:
    - seq = seq_len aufeinanderfolgende Zeitschritte (Voltage, Current, SOC)
    - label = SOC am Schritt (idx + seq_len)
    """
    def __init__(self, df, seq_len=60):
        data_array = df[["Scaled_Voltage[V]", "Scaled_Current[A]", "SOC_ZHU"]].values
        self.data_array = data_array
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data_array) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.data_array[idx : idx + self.seq_len]        # shape (seq_len, 3)
        y_val = self.data_array[idx + self.seq_len, 2]           # SOC_ZHU in Spalte 2
        x_seq_t = torch.tensor(x_seq, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        return x_seq_t, y_val_t

###############################################################################
# 3) LSTMSOCModel mit parametrischen Hyperparametern
###############################################################################
class LSTMSOCModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=1, dropout=0.0, batch_first=True):
        """
        input_size  = Anzahl Features pro Zeitschritt (hier: 3)
        hidden_size = Dimension des LSTM-Hidden-States
        num_layers  = Anzahl gestackter LSTM-Layer
        dropout     = Dropout zwischen LSTM-Layern (nur > 0 wenn num_layers>1)
        """
        super().__init__()
        # ForecastingLSTM aus pytorch_forecasting
        self.lstm = ForecastingLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,    # Pass dropout here
            batch_first=batch_first
        )

        # Einfacher Linear-Layer für 1-D-Output (SOC)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = lstm_out[:, -1, :]    # (batch_size, hidden_size)
        soc_pred = self.fc(last_out)     # (batch_size, 1)
        return soc_pred.squeeze(-1)      # -> (batch_size,)


###############################################################################
# 4) Training-/Validierungs-Helpers
###############################################################################
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    losses = []
    for x_batch, y_batch in tqdm(dataloader, desc="Training Batches"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            losses.append(loss.item())
    return np.mean(losses)

###############################################################################
# 5) Optuna-Objective-Funktion mit TimeSeriesSplit
###############################################################################
def objective(trial):
    # Hyperparameter-Suche
    hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128])
    num_layers  = trial.suggest_int('num_layers', 1, 3)
    dropout     = trial.suggest_float('dropout', 0.0, 0.5)
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay= trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    seq_len     = trial.suggest_categorical('seq_length', [30, 60, 120])
    batch_size  = trial.suggest_categorical('batch_size', [16, 32, 64])
    # Wir setzen die Epochen klein, damit das Tuning nicht zu lange dauert
    max_epochs_in_optuna = trial.suggest_int('epochs', 5, 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wir machen TimeSeriesSplit auf den df_trainval-Daten
    # Für Cross-Validation
    n_splits = 3  # Anzahl Folds
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Um in Numpy-Arrays zu arbeiten, erstellen wir uns ein Zwischenergebnis
    # (die SequenceDataset-Klasse braucht ein df. Wir teilen es also beim TSCV in Abschnitte.)
    df_trainval_reset = df_trainval.reset_index(drop=True)  # Wichtig, damit Index sauber von 0..N geht
    # --> df_trainval_reset hat Spalten "Scaled_Voltage[V]", "Scaled_Current[A]", "SOC_ZHU"
    # plus "Absolute_Time..." und evtl. "timestamp", die wir hier nicht zwingend brauchen

    # MSE pro Fold
    fold_mses = []

    indices = np.arange(len(df_trainval_reset))
    for train_idx, val_idx in tscv.split(indices):
        # Baue DFs für diesen Fold
        df_fold_train = df_trainval_reset.iloc[train_idx]
        df_fold_val   = df_trainval_reset.iloc[val_idx]

        # Erstelle Datasets
        train_ds = SequenceDataset(df_fold_train, seq_len=seq_len)
        val_ds   = SequenceDataset(df_fold_val,   seq_len=seq_len)

        if len(train_ds) <= 0 or len(val_ds) <= 0:
            # Kann passieren, wenn Daten sehr wenige sind und TSCV die Splits zu klein macht.
            # Dann überspringen oder setze MSE sehr hoch
            return 1e9

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)

        # Modell + Optimizer
        model = LSTMSOCModel(
            input_size=3,  # Wir haben 3 Features
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        # Trainiere x Epochen
        for ep in tqdm(range(max_epochs_in_optuna), desc="Optuna Epochs"):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            # optional könnte man pro Epoche validieren, wir machen's hier am Ende

        # Letzter Validation-Loss nach den Epochen
        val_mse = validate_one_epoch(model, val_loader, criterion, device)
        fold_mses.append(val_mse)

    # Durchschnittlicher MSE über alle Folds
    mean_mse = np.mean(fold_mses)

    # CSV logging
    csv_path = Path("optuna_results.csv")
    write_header = not csv_path.exists()
    results = {
        "trial_number": trial.number,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "seq_length": seq_len,
        "batch_size": batch_size,
        "epochs": max_epochs_in_optuna,
        "mean_mse": mean_mse
    }
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(results)

    return mean_mse

###############################################################################
# 6) Optuna-Studie ausführen
###############################################################################
if __name__ == "__main__":
    print("\nStarte Optuna-Studie (TimeSeriesSplit auf Train/Val)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15, timeout=None)  # z.B. 15 Trials

    print("\n--- Optuna Study abgeschlossen ---")
    print(f"Beste gefundene MSE: {study.best_trial.value}")
    print("Beste Parameter:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # 6a) Finales Retraining mit den besten Parametern:
    best_params = study.best_trial.params
    best_hidden_size = best_params['hidden_size']
    best_num_layers  = best_params['num_layers']
    best_dropout     = best_params['dropout']
    best_lr          = best_params['lr']
    best_weight_decay= best_params['weight_decay']
    best_seq_len     = best_params['seq_length']
    best_batch_size  = best_params['batch_size']
    best_epochs      = best_params['epochs']

    # Wir trainieren jetzt auf dem **gesamten** df_trainval
    print("\n--- Starte finales Training mit den besten Parametern ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_trainval_ds = SequenceDataset(df_trainval, seq_len=best_seq_len)
    final_trainval_loader = DataLoader(final_trainval_ds, batch_size=best_batch_size, shuffle=True, drop_last=True)

    final_model = LSTMSOCModel(
        input_size=3,
        hidden_size=best_hidden_size,
        num_layers=best_num_layers,
        dropout=best_dropout,
        batch_first=True
    ).to(device)

    final_optimizer = optim.Adam(final_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
    final_criterion = nn.MSELoss()

    # Wir trainieren mit best_epochs
    for ep in tqdm(range(best_epochs), desc="Final Retrain Epochs"):
        train_mse = train_one_epoch(final_model, final_trainval_loader, final_criterion, final_optimizer, device)
        # optional: Kein eigenes val_loader mehr, da wir alles zusammen benutzen.
        print(f"Final Retrain Ep {ep+1}/{best_epochs} - MSE: {train_mse:.6f}")

    print("\n--- Finales Modell ist trainiert. Testphase folgt. ---")

    ###############################################################################
    # 7) Testphase: Autoregressive Vorhersage (unverändert aus deinem Code)
    ###############################################################################
    def predict_autoregressive(model, df, seq_len=60):
        model.eval()
        data_array = df[["Scaled_Voltage[V]", "Scaled_Current[A]", "SOC_ZHU"]].values
        data_clone = data_array.copy()
        preds = np.full(len(data_clone), np.nan)

        with torch.no_grad():
            for i in range(seq_len, len(data_clone)):
                input_seq = data_clone[i - seq_len : i]
                x_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
                pred_soc = model(x_t).item()
                preds[i] = pred_soc
                data_clone[i, 2] = pred_soc  # SOC überschreiben

        return preds

    # Autoregressive Vorhersage auf dem Test-Set
    preds_test = predict_autoregressive(final_model, df_test, seq_len=best_seq_len)

    # Plot
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
    current_dir = Path(__file__).parent
    models_dir = current_dir / "models"
    os.makedirs(models_dir, exist_ok=True)
    plot_file = models_dir / "prediction_test.png"
    plt.savefig(plot_file)
    print(f"Plot gespeichert unter: {plot_file}")
    plt.show()

    # Modell speichern
    model_path = models_dir / "lstm_soc_model.pth"
    torch.save(final_model.state_dict(), model_path)
    print(f"Modell gespeichert unter: {model_path}")

