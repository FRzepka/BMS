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
    # neue Trainingszellen und feste Validierungszelle
    train_cells = [
        f"MGFarm_18650_C{str(i).zfill(2)}"
        for i in [1,3,5,9,11,13,15,17,19,21,23,25,27]
    ]
    val_cell = "MGFarm_18650_C07"
    # Feature-Liste
    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m"]

    # trainings-Daten initial (nur timestamp ergänzen)
    train_dfs = {}
    for name in train_cells:
        df = cells[name].copy()
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        train_dfs[name] = df

    # scaler auf *allen* Zellen fitten (nicht nur Training)
    df_all = pd.concat(cells.values(), ignore_index=True)
    scaler = MaxAbsScaler().fit(df_all[feats])
    print("[INFO] Skaler über alle Zellen fitten")

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

    print(">> Seq2Seq-Validation startet")
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
    preds = np.array(preds)
    gts = labels[: len(preds)]
    return np.mean((preds - gts) ** 2)

def evaluate_online(model, df, device):
    """Stepwise seq‐to‐seq Validation mit tqdm."""
    model.eval()
    print(">> Online-Validation startet")
    with torch.no_grad():
        for idx, (v, i, soh, qm) in enumerate(zip(
            df['Voltage[V]'], df['Current[A]'],
            df['SOH_ZHU'], df['Q_m']
        )):
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
def train_online(
    epochs=30, lr=1e-4, online_train=False,
    hidden_size=32, dropout=0.2,
    patience=5, log_csv_path="training_log.csv",
    out_dir="trial"
):
    # Ordner für diesen Lauf anlegen
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_csv_path = Path(out_dir) / log_csv_path

    train_scaled, df_val, df_test, train_cells, val_cell = load_data()
    print(f"[INFO] Train cells={train_cells}, Val/Test cell={val_cell}")

    # Rohdaten-Plots
    for name, df in train_scaled.items():
        plt.figure(figsize=(10,4))
        plt.plot(df['timestamp'], df['SOC_ZHU'])
        plt.title(f"Train SOC {name}")
        plt.tight_layout()
        plt.savefig(Path(out_dir)/f"train_{name}.png")
        plt.close()

    model = build_model(hidden_size=hidden_size, dropout=dropout)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optim, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=(device.type=="cuda"))
    best_val = float('inf'); no_improve=0
    log_rows=[]

    for ep in range(1, epochs+1):
        print(f"\n--- Epoch {ep}/{epochs} ---")
        model.train()
        total_loss=0; steps=0

        for name, df in train_scaled.items():
            print(f"[Epoch {ep}] Training Cell {name}")
            if not online_train:
                ds = CellDataset(df, SEQ_CHUNK_SIZE)
                dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
                h, c = init_hidden(model, device=device)
                for x_b, y_b in dl:
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
                    steps += 1
            else:
                print(f"[Epoch {ep}] Online-Training Cell {name}")
                h, c = init_hidden(model, device=device)
                for v, i, soh, qm, y_true in zip(
                    df['Voltage[V]'], df['Current[A]'],
                    df['SOH_ZHU'], df['Q_m'], df['SOC_ZHU']
                ):
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
                    steps += 1

        train_rmse = math.sqrt(total_loss/steps)
        print(f"[Epoch {ep}] train RMSE={train_rmse:.4f}")

        # nur MSE berechnen, kein Plot mehr
        val_mse, _, _ = evaluate_onechunk_seq2seq(model, df_val, device)
        val_rmse = math.sqrt(val_mse)
        print(f"[Epoch {ep}] val RMSE={val_rmse:.4f}")

        scheduler.step(val_mse)
        if val_mse < best_val:
            best_val = val_mse; no_improve=0
            torch.save(model.state_dict(), Path(out_dir)/"best_model.pth")
        else:
            no_improve +=1
            if no_improve>=patience:
                print(f"[INFO] Early stopping nach {patience} Epochen")
                break

        log_rows.append({"epoch":ep, "train_rmse":train_rmse, "val_rmse":val_rmse})
        with open(log_csv_path, "w", newline="") as f:
            writer=csv.DictWriter(f, fieldnames=["epoch","train_rmse","val_rmse"])
            writer.writeheader(); writer.writerows(log_rows)

    return train_rmse, val_rmse

def test_seq2seq(log_csv_path="training_log.csv", out_dir="trial"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    _, _, df_test, _, val_cell = load_data()
    print(f"[TEST] Zelle {val_cell}")
    model = build_model()
    model.load_state_dict(torch.load(Path(out_dir)/"best_model.pth", map_location=device))
    model.eval()

    mse, preds, gts = evaluate_onechunk_seq2seq(model, df_test, device)
    rmse = math.sqrt(mse); mae=np.mean(np.abs(preds-gts))
    print(f"[TEST] RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Test-Plot mit MAE-Annotation
    plt.figure(figsize=(10,4))
    plt.plot(df_test['timestamp'], gts, 'k-', label="GT")
    plt.plot(df_test['timestamp'], preds, 'r-', label="Pred")
    plt.title(f"Final Test — MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    plt.legend(loc="best")
    plt.annotate(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}", xy=(0.01,0.95), xycoords='axes fraction', va='top')
    plt.tight_layout()
    plt.savefig(Path(out_dir)/"test_plot.png"); plt.close()

    # Test-Log anhängen
    try:
        with open(Path(out_dir)/log_csv_path, "a", newline="") as f:
            writer=csv.writer(f)
            writer.writerow([]); writer.writerow(["test_mae","test_rmse"]); writer.writerow([mae, rmse])
    except Exception as e:
        print(f"[WARN] Konnte Test-Log nicht schreiben: {e}")

    return mae, rmse

def objective(trial):
    hs = trial.suggest_int("hidden_size",32,256)
    dr = trial.suggest_float("dropout",0.0,0.5)
    lr = trial.suggest_loguniform("lr",1e-5,1e-3)
    out_dir = f"trial_{trial.number}"
    train_rmse, val_rmse = train_online(
        epochs=30, lr=lr, online_train=False,
        hidden_size=hs, dropout=dr,
        patience=5, log_csv_path="log.csv",
        out_dir=out_dir
    )
    test_mae, test_rmse = test_seq2seq(log_csv_path="log.csv", out_dir=out_dir)
    print(f"[TRIAL {trial.number}] val_rmse={val_rmse:.4f}, test_mae={test_mae:.4f}, test_rmse={test_rmse:.4f}")
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