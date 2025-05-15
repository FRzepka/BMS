import os
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
import math
import itertools
import optuna

# Konstanten
SEQ_CHUNK_SIZE = 4096    # Länge der Zeitreihen-Chunks für Seq-to-Seq

# Gerät auswählen und cuDNN optimieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Datenlade-Funktion
def load_cell_data(data_dir: Path):
    dataframes = {}
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("MGFarm_18650_"):
            dfp = folder / "df.parquet"
            if dfp.exists():
                dataframes[folder.name] = pd.read_parquet(dfp)
            else:
                print(f"Warning: {dfp} fehlt")
    return dataframes

# Daten vorbereiten
def load_data(base_path: str = "/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes"):
    base = Path(base_path)
    cells = load_cell_data(base)
    # Festgelegte Trainings- und Validierungszellen
    train_cells = [
        "MGFarm_18650_C05",
        "MGFarm_18650_C01",
        "MGFarm_18650_C21",
        "MGFarm_18650_C19"
    ]
    val_cell = "MGFarm_18650_C07"

    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m"]  # include Q_m
    # Trainingsdaten laden und Timestamp
    train_dfs = {}
    for name in train_cells:
        df = cells[name].copy()
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        train_dfs[name] = df

    # Skalar fitten (Min-Max zwischen 0 und 1)
    df_all_train = pd.concat(train_dfs.values(), ignore_index=True)
    scaler = MaxAbsScaler().fit(df_all_train[feats])
    # debug: inspect scaler learned max_abs per feature
    print("[DEBUG] scaler.max_abs_:", dict(zip(feats, scaler.scale_)))

    # Skalierte Trainingsdaten
    train_scaled = {}
    for name, df in train_dfs.items():
        df2 = df.copy()
        df2[feats] = scaler.transform(df2[feats])
        train_scaled[name] = df2
    # debug: check for NaNs after scaling
    for name, df2 in train_scaled.items():
        nan_counts = pd.DataFrame(df2[feats]).isna().sum().to_dict()
        print(f"[DEBUG] {name} NaNs after train scaling:", {k:v for k,v in nan_counts.items() if v>0} or "none")

    # Validierung/Test der dritten Zelle
    df3 = cells[val_cell].copy()
    df3['timestamp'] = pd.to_datetime(df3['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    L = len(df3)
    # 50% der Zelle für Val, die nächsten 10% für Test
    i1, i2 = int(L * 0.5), int(L * 0.6)
    df_val  = df3.iloc[:i1].copy()
    df_test = df3.iloc[i1:i2].copy()
    df_val[feats]  = scaler.transform(df_val[feats])
    df_test[feats] = scaler.transform(df_test[feats])
    # debug: shapes & NaNs in val/test
    print(f"[DEBUG] df_val length: {len(df_val)}, df_test length: {len(df_test)}")
    print("[DEBUG] df_val NaNs:", df_val[feats].isna().sum().to_dict())
    print("[DEBUG] df_test NaNs:", df_test[feats].isna().sum().to_dict())

    return train_scaled, df_val, df_test, train_cells, val_cell

# Angepasstes Dataset für ganze Zellen
class CellDataset(Dataset):
    def __init__(self, df, sequence_length=SEQ_CHUNK_SIZE):
        """Dataset für eine ganze Zelle, aufgeteilt in Sequenz-Chunks"""
        self.sequence_length = sequence_length
        self.data   = df[["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m"]].values
        self.labels = df["SOC_ZHU"].values
        self.n_batches = max(1, len(self.data) // self.sequence_length)
    
    def __len__(self):
        return self.n_batches  # Anzahl der Batches
    
    def __getitem__(self, idx):
        start = idx * self.sequence_length
        end = min(start + self.sequence_length, len(self.data))
        x = torch.from_numpy(self.data[start:end]).float()
        y = torch.from_numpy(self.labels[start:end]).float()
        return x, y

# Weight-initialization helper
def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, p in m.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

# Modell: LSTM + Dropout + MLP-Head
def build_model(input_size=4, hidden_size=32, num_layers=1, dropout=0.2, mlp_hidden=32):
    class SOCModel(nn.Module):
        def __init__(self):
            super().__init__()
            # LSTM ohne Dropout (voller Informationsfluss)
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=0.0)
            # hidden_size bestimmt die Dim. der LSTM-Ausgabe
            # mlp_hidden ist die Größe der verborgenen MLP-Schicht
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),   # nur hier Dropout
                nn.Linear(mlp_hidden, 1),
                nn.Sigmoid()
            )

        def forward(self, x, hidden):
            self.lstm.flatten_parameters()       # cuDNN-ready
            x = x.contiguous()                   # ensure input contiguous
            # make hidden states contiguous
            h, c = hidden
            h, c = h.contiguous(), c.contiguous()
            hidden = (h, c)
            out, hidden = self.lstm(x, hidden)
            batch, seq_len, hid = out.size()
            out_flat = out.contiguous().view(batch * seq_len, hid)
            soc_flat = self.mlp(out_flat)
            soc = soc_flat.view(batch, seq_len)
            return soc, hidden
    model = SOCModel().to(device)
    # 2) init weights & optimize cuDNN for multi-layer LSTM
    model.apply(init_weights)
    model.lstm.flatten_parameters()
    return model

# Helper-Funktion für die Initialisierung der hidden states
def init_hidden(model, batch_size=1, device=None):
    if device is None:
        device = next(model.parameters()).device
    h = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size, device=device)
    c = torch.zeros_like(h)
    return h, c

# ——— Neue Seq-to-Seq-Funktion für Validierung/Test —————————————————————————
def evaluate_seq2seq(model, df, device):
    """
    Seq-to-Seq-Validation mit Chunking und TQDM.
    """
    model.eval()
    seq    = df[["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m"]].values
    labels = df["SOC_ZHU"].values
    total = len(seq)
    n_chunks = math.ceil(total / SEQ_CHUNK_SIZE)
    h, c = init_hidden(model, batch_size=1, device=device)
    h, c = h.contiguous(), c.contiguous()
    preds = []

    pbar = tqdm(total=n_chunks, desc="Seq2Seq Val", leave=False)
    with torch.no_grad():
        for i in range(n_chunks):
            s = i * SEQ_CHUNK_SIZE
            e = min(s + SEQ_CHUNK_SIZE, total)
            chunk = torch.tensor(seq[s:e], dtype=torch.float32, device=device).unsqueeze(0)
            chunk = chunk.contiguous()
            model.lstm.flatten_parameters()
            # disable cuDNN here, um lange/sehr große Chunks zu erlauben
            with torch.backends.cudnn.flags(enabled=False):
                out, (h, c) = model(chunk, (h, c))
            h, c = h.contiguous(), c.contiguous()
            preds.extend(out.squeeze(0).cpu().numpy())
    pbar.close()

    preds = np.array(preds)
    gts = labels[: len(preds)]
    return np.mean((preds - gts) ** 2)

def evaluate_online(model, df, device):
    """Stepwise seq‐to‐seq Validation mit tqdm."""
    model.eval()
    h, c = init_hidden(model, device=device)
    preds, gts = [], []
    with torch.no_grad():
        for idx, (v, i, soh, qm) in enumerate(
            tqdm(zip(df['Voltage[V]'].values,
                     df['Current[A]'].values,
                     df['SOH_ZHU'].values,
                     df['Q_m'].values),
                 total=len(df), desc="Validation", leave=False)
        ):
            x = torch.tensor([[v, i, soh, qm]], dtype=torch.float32, device=device).view(1,1,4).contiguous()
            pred, (h, c) = model(x, (h, c))
            preds.append(pred.item())
            gts.append(df['SOC_ZHU'].iloc[idx])
    preds, gts = np.array(preds), np.array(gts)
    return np.mean((preds - gts)**2)

def evaluate_onechunk_seq2seq(model, df, device):
    """
    Seq2Seq-Eval über genau einen Chunk: ganzes df als (1, N, F)-Sequenz.
    """
    model.eval()
    seq    = df[["Voltage[V]","Current[A]","SOH_ZHU","Q_m"]].values
    labels = df["SOC_ZHU"].values
    h, c   = init_hidden(model, batch_size=1, device=device)
    # ensure hidden states contiguous
    h, c   = h.contiguous(), c.contiguous()
    chunk  = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
    # ensure input contiguous
    chunk  = chunk.contiguous()
    with torch.no_grad():
        model.lstm.flatten_parameters()
        # disable cuDNN hier, um sehr lange Ein-Chuck-Sequenzen zu erlauben
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = model(chunk, (h, c))
    preds = out.squeeze(0).cpu().numpy()
    mse   = np.mean((preds - labels)**2)
    return mse, preds, labels

# Training Funktion mit Batch-Training und Seq2Seq-Validierung
def train_online(epochs=30, lr=1e-4, online_train=False,
                 hidden_size=32, dropout=0.2,
                 patience=5, log_csv_path="training_log.csv"):
    train_scaled, df_val, df_test, train_cells, val_cell = load_data()
    # debug: summary of data splits
    print(f"[DEBUG] Chunk size: {SEQ_CHUNK_SIZE}")
    for name, df in train_scaled.items():
        print(f"[DEBUG] TRAIN {name}: {len(df)} rows")
    print(f"[DEBUG] VALIDATION ({val_cell}): {len(df_val)} rows")
    print(f"[DEBUG] TEST       ({val_cell}): {len(df_test)} rows")
    print("Training auf Zellen:", train_cells)

    # Rohdaten-Plots
    for name, df in train_scaled.items():
        plt.figure(figsize=(10,4))
        plt.plot(df['timestamp'], df['SOC_ZHU'], label=name)
        plt.title(f"Train SOC {name}")
        plt.tight_layout(); plt.savefig(f"train_{name}_plot.png"); plt.close()

    # Val-Plot
    plt.figure(figsize=(8,4))
    plt.plot(df_val['timestamp'], df_val['SOC_ZHU'])
    plt.title("Val SOC")
    plt.tight_layout(); plt.savefig("val_data_plot.png"); plt.close()

    # Test-Plot
    plt.figure(figsize=(8,4))
    plt.plot(df_test['timestamp'], df_test['SOC_ZHU'])
    plt.title("Test SOC")
    plt.tight_layout(); plt.savefig("test_data_plot.png"); plt.close()

    model = build_model(hidden_size=hidden_size, dropout=dropout)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optim, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    best_val_loss = float('inf')
    no_improve = 0

    # --- Logging vorbereiten ---
    log_fields = ["epoch", "train_rmse", "val_rmse"]
    log_rows = []

    for ep in range(1, epochs+1):
        model.train()
        total_loss, total_steps = 0.0, 0
        print(f"\n=== Epoch {ep}/{epochs} ===")

        for name, df in train_scaled.items():
            if not online_train:
                # Standard batch training - same as 1.2.3.2
                ds = CellDataset(df, SEQ_CHUNK_SIZE)
                print(f"--> {name}, Batches: {len(ds)}")
                dl = DataLoader(
                    ds,
                    batch_size=1,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
                h, c = init_hidden(model, device=device)
                h, c = h.contiguous(), c.contiguous()  # Ensure contiguous hidden states
                
                for x_b, y_b in tqdm(dl, desc=f"{name} Ep{ep}", leave=True):
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    x_b = x_b.contiguous()  # Ensure contiguous input
                    
                    optim.zero_grad()
                    
                    # Use proper precision context
                    with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                        model.lstm.flatten_parameters()  # Optimize LSTM
                        pred, (h, c) = model(x_b, (h, c))
                        loss = criterion(pred, y_b)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optim)
                    scaler.update()
                    
                    h, c = h.detach(), c.detach()
                    total_loss += loss.item()   
                    total_steps += 1
            else:
                # Online training - preserved from 1.2.3.2
                print(f"--> {name} (Online-Train, {len(df)} steps)")
                h, c = init_hidden(model, batch_size=1, device=device)
                h, c = h.contiguous(), c.contiguous()
                
                for idx, (v, i, soh, y_true) in enumerate(
                    tqdm(zip(df['Voltage[V]'].values, df['Current[A]'].values, df['SOH_ZHU'].values, df['Q_m'].values, df['SOC_ZHU'].values),
                         total=len(df), desc=f"{name} Ep{ep}", leave=True)):
                    x = torch.tensor([[v, i, soh, qm]], dtype=torch.float32, device=device).view(1,1,4).contiguous()
                    y = torch.tensor([[y_true]], dtype=torch.float32, device=device).contiguous()
                    optim.zero_grad()
                    
                    with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                        model.lstm.flatten_parameters()
                        pred, (h, c) = model(x, (h, c))
                        loss = criterion(pred, y)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optim)
                    scaler.update()
                    
                    h, c = h.detach(), c.detach()
                    total_loss += loss.item()                    
                    total_steps += 1

        avg_mse = total_loss / total_steps
        train_rmse = math.sqrt(avg_mse)
        print(f"Epoch {ep} Training abgeschlossen, train RMSE={train_rmse:.6f}")

        # → Ein-Chunk-Validation
        val_mse, val_preds, val_gts = evaluate_onechunk_seq2seq(model, df_val, device)
        val_rmse = math.sqrt(val_mse)
        print(f"Epoch {ep} Validierung abgeschlossen, val RMSE={val_rmse:.6f}")
        plt.figure(figsize=(10,4))
        plt.plot(df_val['timestamp'], val_gts,  'k-', label="GT")
        plt.plot(df_val['timestamp'], val_preds,'r-', label="Pred")
        plt.title(f"Validierung Ep{ep} — RMSE: {val_rmse:.4f}")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"val_onechunk_epoch{ep:02d}.png"); plt.close()

        scheduler.step(val_mse)
        # EarlyStopping
        if val_mse < best_val_loss:
            best_val_loss, no_improve = val_mse, 0
            torch.save(model.state_dict(), "best_seq2seq_soc.pth")
        else:
            no_improve +=1
        if no_improve >= patience:
            print(f"Early stopping nach {patience} Epochen ohne Verbesserung")
            break

        # --- Logging: Zeile anhängen ---
        log_rows.append({"epoch": ep, "train_rmse": train_rmse, "val_rmse": val_rmse})

        # --- Logging: CSV nach jeder Epoche aktualisieren ---
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
            writer.writerows(log_rows)

    # Ende von train_online: return metrics für Optuna
    return train_rmse, val_rmse

# Test: Seq-to-Seq-Inferenz mit einem Forward-Pass
def test_seq2seq(log_csv_path="training_log.csv"):
    _, _, df_test, _, val_cell = load_data()
    print("Test auf Zelle:", val_cell)
    
    model = build_model()
    model.load_state_dict(torch.load("best_seq2seq_soc.pth", map_location=device))    
    model.eval()

    # Ein-Chunk-Test
    test_mse, test_preds, test_gts = evaluate_onechunk_seq2seq(model, df_test, device)
    test_rmse = math.sqrt(test_mse)
    test_mae  = np.mean(np.abs(test_preds - test_gts))
    print(f"Finaler Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    plt.figure(figsize=(10,4))
    plt.plot(df_test['timestamp'], test_gts,   'k-', label="GT")
    plt.plot(df_test['timestamp'], test_preds, 'r-', label="Pred")
    plt.title(f"Finaler Test — MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    plt.legend(); plt.tight_layout()
    plt.savefig("test_onechunk.png"); plt.close()

    # --- Logging: Testresultate an CSV anhängen ---
    try:
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["test_mae", "test_rmse"])
            writer.writerow([test_mae, test_rmse])
    except Exception as e:
        print(f"Fehler beim Schreiben der Testresultate in die CSV: {e}")

    # Plots
    plt.figure(figsize=(10,4))
    plt.plot(timestamps, gts, 'k-', label="GT")
    plt.plot(timestamps, preds, 'r-', label="Pred")
    plt.title("Seq2Seq Final Test")
    plt.legend()
    plt.annotate(f"MAE: {test_mae:.4f}\nRMSE: {test_rmse:.4f}", xy=(0.01,0.95), xycoords='axes fraction', va='top')
    plt.tight_layout(); plt.savefig("final_seq2seq_plot.png"); plt.close()

    zoom_n = min(50000, len(preds))
    for name_seg, seg in [("Start", slice(0, zoom_n)), ("End", slice(-zoom_n, None))]:
        plt.figure(figsize=(10,4))
        plt.plot(timestamps[seg], gts[seg], 'k-', label="GT")
        plt.plot(timestamps[seg], preds[seg], 'r-', label="Pred")
        plt.legend()
        plt.title(f"Zoom {name_seg}")
        plt.tight_layout()
        plt.savefig(f"zoom_{name_seg.lower()}_seq2seq_plot.png")
        plt.close()

# Optuna-Studie mit Objective
def objective(trial):
    # Sample Hyperparameter
    hs  = trial.suggest_int("hidden_size", 32, 256)
    dr  = trial.suggest_float("dropout", 0.0, 0.5)
    lr  = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    # Train & Val
    train_rmse, val_rmse = train_online(
        epochs=30, lr=lr, online_train=False,
        hidden_size=hs, dropout=dr,
        patience=5, log_csv_path=f"optuna_{hs}_{dr}_{lr:.0e}.csv"
    )
    return val_rmse

def tune_optuna(n_trials: int = 20):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best Params:", study.best_params)
    # Nach Optuna: finaler Test
    hs, dr, lr = study.best_params["hidden_size"], study.best_params["dropout"], study.best_params["lr"]
    _, _ = train_online(epochs=30, lr=lr, online_train=False,
                        hidden_size=hs, dropout=dr,
                        patience=5, log_csv_path="best_optuna.csv")
    test_rmse, test_mae = test_seq2seq(log_csv_path="best_optuna.csv")
    print(f"Final Test RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

if __name__ == "__main__":
    tune_optuna(n_trials=20)