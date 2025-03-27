import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import optuna
from tqdm import tqdm  # Für Fortschrittsbalken
import shutil

torch.cuda.empty_cache()

# Ordner für Hyperparametertuning-Ergebnisse (HPT)
hpt_folder = Path(__file__).parent / "HPT" if '__file__' in globals() else Path("HPT")
os.makedirs(hpt_folder, exist_ok=True)

###############################################################################
# 1) Laden der Daten, Reduktion auf 25%, zeitbasierter Split in [Train, Val, Test]
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
            print(f"[INFO] Loaded {folder.name}")
        else:
            print(f"[WARN] No df.parquet found in {folder.name}")
    else:
        print("[WARN] Folder MGFarm_18650_C01 not found")
    return dataframes

# Passe den Speicherort an
data_dir = Path('/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes')
cell_data = load_cell_data(data_dir)

cell_keys = sorted(cell_data.keys())[:1]
if len(cell_keys) < 1:
    raise ValueError("Keine Zelle gefunden; bitte prüfen.")

cell_name = cell_keys[0]  # z.B. 'C01'
df_full = cell_data[cell_name]

# Verwende 25% der Daten
sample_size = int(len(df_full) * 0.25)
df_small = df_full.head(sample_size).copy()
print(f"[INFO] Gesamtdaten: {len(df_full)}, verwende 25% = {sample_size} Zeilen.")

# Konvertiere Timestamp (Daten liegen in Sekunden)
df_small['timestamp'] = pd.to_datetime(df_small['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])

# Zeitbasiertes Split (Train: 40%, Val: 40%, Test: 20%)
len_small = len(df_small)
train_end = int(len_small * 0.4)
val_end   = int(len_small * 0.8)

df_train = df_small.iloc[:train_end]
df_val   = df_small.iloc[train_end:val_end]
df_test  = df_small.iloc[val_end:]
print(f"[INFO] Split: Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

###############################################################################
# 2) Skalierung von Voltage & Current
###############################################################################
scaler = StandardScaler()
features_to_scale = ['Voltage[V]', 'Current[A]']

# Fit nur auf Trainingsdaten
scaler.fit(df_train[features_to_scale])

df_train_scaled = df_train.copy()
df_val_scaled   = df_val.copy()
df_test_scaled  = df_test.copy()

df_train_scaled[features_to_scale] = scaler.transform(df_train_scaled[features_to_scale])
df_val_scaled[features_to_scale]   = scaler.transform(df_val_scaled[features_to_scale])
df_test_scaled[features_to_scale]  = scaler.transform(df_test_scaled[features_to_scale])

###############################################################################
# 3) Dataset-Klasse (seq2one, NICHT autoregressiv, SOC nur als Label)
###############################################################################
class SequenceDataset(Dataset):
    """
    Für jedes Sample:
      - Input: Fenster aus [Voltage, Current] über seq_len Sekunden
      - Label: SOC an (t + seq_len) (immer 1 Schritt voraus)
    """
    def __init__(self, df, seq_len=60):
        self.seq_len = seq_len
        self.features = df[["Voltage[V]", "Current[A]"]].values  # shape=(N, 2)
        self.labels   = df["SOC_ZHU"].values                     # shape=(N,)

    def __len__(self):
        return len(self.labels) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.features[idx : idx + self.seq_len]   # shape: (seq_len, 2)
        y_val = self.labels[idx + self.seq_len]             # Vorhersage: 1 Schritt
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

###############################################################################
# 4) TCN-Modell und Hilfsklassen
###############################################################################
class Chomp1d(nn.Module):
    """
    Schneidet das "causal padding" beim 1D-Convolution ab.
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x shape: (batch_size, out_channels, seq_len + padding)
        return x[:, :, :-self.chomp_size]

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        self.chomp1 = Chomp1d(chomp_size=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        self.chomp2 = Chomp1d(chomp_size=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # Residual Connection; falls in_channels != out_channels, Downsampling
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if (in_channels != out_channels) else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """
    Einfaches TCN bestehend aus einer Liste von TCN-Blöcken.
    - input_size: Anzahl der Features (hier 2: Voltage, Current)
    - num_channels: Liste der Kanalzahlen pro Block (z. B. [32, 32])
    - kernel_size und dropout sind hyperparametrierbar.
    """
    def __init__(self, input_size=2, num_channels=[32, 32], kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            # Padding = (kernel_size - 1) * dilation_size
            padding = (kernel_size - 1) * dilation_size

            block = TCNBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=padding,
                dropout=dropout
            )
            layers.append(block)

        self.tcn = nn.Sequential(*layers)
        self.fc  = nn.Linear(num_channels[-1], 1)  # finale Fully Connected-Schicht

    def forward(self, x):
        """
        x kommt als (batch_size, seq_len, input_size) rein.
        Wir permutieren zu (batch_size, input_size, seq_len), da Conv1d dies erwartet.
        """
        x = x.permute(0, 2, 1)  # => shape: (batch_size, 2, seq_len)
        y = self.tcn(x)         # => shape: (batch_size, out_channels, seq_len)
        # Nimm den letzten Zeitschritt für die 1-Schritt Vorhersage
        last_out = y[:, :, -1]
        out = self.fc(last_out)
        return out.squeeze(-1)

###############################################################################
# 5) Optuna Objective-Funktion für das Hyperparametertuning mit tqdm
###############################################################################
def objective(trial: optuna.Trial):
    # Hyperparameter
    seq_length = trial.suggest_int("seq_length", 60, 900, step=30)  # in Sekunden
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    n_ch1 = trial.suggest_int("n_ch1", 16, 128, step=16)
    n_ch2 = trial.suggest_int("n_ch2", 16, 128, step=16)
    num_channels = [n_ch1, n_ch2]
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)

    print(f"\n[TRIAL {trial.number}] Beginne mit: seq_length={seq_length}, lr={lr:.5f}, "
          f"kernel_size={kernel_size}, dropout={dropout:.2f}, num_channels={num_channels}, "
          f"batch_size={batch_size}")

    # Erstelle Datasets
    train_dataset = SequenceDataset(df_train_scaled, seq_len=seq_length)
    val_dataset   = SequenceDataset(df_val_scaled, seq_len=seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Modell instanziieren
    model = TCN(input_size=2, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_epochs = 10
    best_trial_val = float('inf')
    best_model_state = None

    epoch_pbar = tqdm(range(n_epochs), desc=f"Trial {trial.number} Epochs", leave=False)
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        # Trainingsschleife mit Fortschrittsbalken
        for x_batch, y_batch in tqdm(train_loader, desc="Train Batches", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in tqdm(val_loader, desc="Val Batches", leave=False):
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_pred_val = model(x_val)
                loss_val = criterion(y_pred_val, y_val)
                val_loss += loss_val.item() * x_val.size(0)
        val_loss /= len(val_loader.dataset)

        epoch_pbar.set_postfix({"train_loss": f"{train_loss:.6f}", "val_loss": f"{val_loss:.6f}"})

        # Speichere bestes Modell im Trial
        if val_loss < best_trial_val:
            best_trial_val = val_loss
            best_model_state = model.state_dict()

        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f"[TRIAL {trial.number}] Pruned at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

    if best_model_state is not None:
        model_file = hpt_folder / f"model_trial_{trial.number}.pth"
        torch.save(best_model_state, model_file)
        trial.set_user_attr("model_path", str(model_file))
        print(f"[TRIAL {trial.number}] Bestes Modell gespeichert unter: {model_file}")

    torch.cuda.empty_cache()
    print(f"[TRIAL {trial.number}] Abgeschlossen mit val_loss: {best_trial_val:.6f}")
    return best_trial_val

###############################################################################
# 6) Optuna-Studie starten und Ergebnisse speichern
###############################################################################
if __name__ == '__main__':
    print("[INFO] Starte Hyperparametertuning mit Optuna ...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("\n[INFO] Beste Trial:")
    best_trial = study.best_trial
    print(f"  Trial {best_trial.number} | Value: {best_trial.value:.6f}")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Speichere den Modell-State des besten Trials als best_tcn_soc_model.pth
    best_model_path = best_trial.user_attrs.get("model_path", None)
    if best_model_path is not None:
        final_model_path = hpt_folder / "best_tcn_soc_model.pth"
        shutil.copy(best_model_path, final_model_path)
        print(f"[INFO] Bestes Modell wurde gespeichert unter: {final_model_path}")
    else:
        print("[WARN] Kein Modellpfad im besten Trial gefunden!")

    # Speichere alle Trial-Informationen in einer Excel-Datei
    records = []
    for trial in study.trials:
        record = {
            "trial_number": trial.number,
            "value": trial.value,
            "state": trial.state.name,
        }
        record.update(trial.params)
        record["model_path"] = trial.user_attrs.get("model_path", "")
        records.append(record)

    df_trials = pd.DataFrame(records)
    excel_file = hpt_folder / "hpt_trials.xlsx"
    df_trials.to_excel(excel_file, index=False)
    print(f"[INFO] Alle Trial-Informationen wurden in '{excel_file}' gespeichert.")
