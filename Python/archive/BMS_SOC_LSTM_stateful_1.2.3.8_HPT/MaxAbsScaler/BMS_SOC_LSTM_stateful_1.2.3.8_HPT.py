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
import csv
import math
import itertools
import optuna
import pickle  

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

    return train_scaled, df_val, df_test, train_cells, val_cell, scaler

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
    # initialize hidden state and result lists
    h, c = init_hidden(model, batch_size=1, device=device)
    preds, gts = [], []
    with torch.no_grad():
        for idx, (v, i, soh, qm, y_true) in enumerate(zip(
            df['Voltage[V]'], df['Current[A]'],
            df['SOH_ZHU'], df['Q_m'], df['SOC_ZHU']
        )):
            x = torch.tensor([[v, i, soh, qm]], dtype=torch.float32, device=device).view(1,1,4).contiguous()
            pred, (h, c) = model(x, (h, c))
            preds.append(pred.item())
            gts.append(y_true)
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
    out_dir="trial" , train_data=None, df_val=None, df_test=None, feature_scaler=None):

    # convert out_dir to Path so "/" operator works
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_csv_path = out_dir / log_csv_path

    if train_data is None:
        train_scaled, df_val, df_test, train_cells, val_cell, feature_scaler = load_data()
    else:
        train_scaled = train_data
        # reuse globals
        train_cells = train_cells_glob
        val_cell    = val_cell_glob
        # passed dfs/scaler
        feature_scaler = feature_scaler
    print(f"[INFO] Train cells={train_cells}, Val/Test cell={val_cell}")

    model = build_model(hidden_size=hidden_size, dropout=dropout)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optim, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    gradient_scaler = GradScaler(enabled=(device.type=="cuda"))
    best_val = float('inf'); no_improve = 0

    # HISTORY & LOG INITIALIZATION
    train_rmse_history = []
    val_rmse_history   = []
    log_rows = []  # <-- ensure log_rows exists

    for ep in range(1, epochs+1):
        print(f"\n--- Epoch {ep}/{epochs} ---")
        model.train()
        total_loss, steps = 0, 0

        for name, df in train_scaled.items():
            print(f"[Epoch {ep}] Training Cell {name}")
            if not online_train:
                ds = CellDataset(df, SEQ_CHUNK_SIZE)
                dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
                h, c = init_hidden(model, batch_size=1, device=device)
                for x_b, y_b in dl:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    x_b = x_b.contiguous()  # Ensure contiguous input
                    
                    optim.zero_grad()
                    
                    # Use proper precision context
                    with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                        model.lstm.flatten_parameters()  # Optimize LSTM
                        pred, (h, c) = model(x_b, (h, c))
                        loss = criterion(pred, y_b)
                    
                    gradient_scaler.scale(loss).backward()
                    gradient_scaler.unscale_(optim)
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    gradient_scaler.step(optim)
                    gradient_scaler.update()
                    
                    h, c = h.detach(), c.detach()
                    total_loss += loss.item()   
                    steps += 1
            else:
                print(f"[Epoch {ep}] Online-Training Cell {name}")
                h, c = init_hidden(model, batch_size=1, device=device)
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
                    
                    gradient_scaler.scale(loss).backward()
                    gradient_scaler.unscale_(optim)
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    gradient_scaler.step(optim)
                    gradient_scaler.update()
                    
                    h, c = h.detach(), c.detach()
                    total_loss += loss.item()                    
                    steps += 1

        train_rmse = math.sqrt(total_loss/steps)
        train_rmse_history.append(train_rmse)
        print(f"[Epoch {ep}] train RMSE={train_rmse:.4f}")

        # nur MSE berechnen, kein Plot mehr
        val_mse, val_preds, val_labels = evaluate_onechunk_seq2seq(model, df_val, device)
        val_rmse = math.sqrt(val_mse)
        val_rmse_history.append(val_rmse)
        print(f"[Epoch {ep}] val RMSE={val_rmse:.4f}")

        # Early Stopping & Model Save
        is_best = val_rmse < best_val
        best_val = min(val_rmse, best_val)
        if is_best:
            no_improve = 0
            # Modell speichern
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            print(f"[Epoch {ep}] Modell gespeichert: {out_dir / 'best_model.pth'}")
            # Plot Validation Predictions
            plt.figure()
            plt.plot(val_labels, label="True SOC")
            plt.plot(val_preds,  label="Predicted SOC")
            plt.legend()
            plt.title(f"Epoch {ep} Best Val SOC Prediction")
            plt.savefig(out_dir / f"val_plot_epoch_{ep}.png")
            plt.close()
        else:
            no_improve += 1

        # Learning Rate anpassen
        scheduler.step(val_mse)

        # Logging
        log_rows.append({
            "epoch": ep,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "lr": optim.param_groups[0]['lr'],
            "hidden_size": hidden_size,
            "dropout": dropout
        })
        df_log = pd.DataFrame(log_rows)
        df_log.to_csv(log_csv_path, index=False)

        # Frühes Stoppen
        if no_improve >= patience:
            print(f"[INFO] Frühes Stoppen bei Epoche {ep}")
            break

    # Lade das beste Modell für die finale Bewertung
    model.load_state_dict(torch.load(out_dir / "best_model.pth"))

    # Finale Bewertung auf Val und Test
    val_mse, val_preds, val_labels = evaluate_onechunk_seq2seq(model, df_val, device)
    test_mse, test_preds, test_labels = evaluate_onechunk_seq2seq(model, df_test, device)
    val_rmse = math.sqrt(val_mse)
    test_rmse = math.sqrt(test_mse)
    print(f"[INFO] Finale Bewertung -> Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

    # PLOT-Verlauf zeichnen
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_rmse_history)+1), train_rmse_history, label="Train RMSE")
    plt.plot(range(1, len(val_rmse_history)+1),   val_rmse_history,   label="Val RMSE")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Trainingsverlauf")
    plt.grid()
    plt.savefig(out_dir / "train_val_rmse_plot.png")
    plt.show()

    return model, feature_scaler, log_rows, df_val, df_test

# Global data loading & scaling once before HPT
train_scaled_glob, df_val_glob, df_test_glob, train_cells_glob, val_cell_glob, feature_scaler_glob = load_data()
print("[INFO] Global data loaded and scaled once for HPT.")

# Optuna-Optimierung für Hyperparameter
def objective(trial):
    # 1) Hyperparam einlesen
    epochs = 30
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    hidden_size = trial.suggest_int("hidden_size", 16, 128, step=16)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)

    # 2) dynamischen Ordnernamen bauen
    num = trial.number
    dr_str = f"{dropout:.4f}"
    lr_str = f"{lr:.0e}"
    folder = f"trial_{num:02d}_hs{hidden_size}_dr{dr_str}_lr{lr_str}"

    # 3) Training
    model, feature_scaler, log_rows, df_val, df_test = train_online(
        epochs=epochs, lr=lr,
        hidden_size=hidden_size, dropout=dropout,
        patience=5, out_dir=folder,
        train_data=train_scaled_glob,
        df_val=df_val_glob, df_test=df_test_glob,
        feature_scaler=feature_scaler_glob
    )
    # zusätzlicher Validation-Plot für diesen Trial
    val_mse, val_preds, val_labels = evaluate_onechunk_seq2seq(model, df_val, device)
    plt.figure()
    plt.plot(val_labels, label="True SOC")
    plt.plot(val_preds,  label="Predicted SOC")
    plt.legend()
    plt.title(f"Trial {trial.number} Val SOC Prediction")
    plt.savefig(Path(folder) / "val_plot.png")
    plt.close()

    # Test-MAE berechnen
    _, test_preds, test_labels = evaluate_onechunk_seq2seq(model, df_test, device)
    test_mae = np.mean(np.abs(test_preds - test_labels))
    Path(folder).joinpath("test_mae.txt").write_text(f"{test_mae:.6f}")
    # Plot und speichern
    plt.figure()
    plt.plot(test_labels, label="True SOC")
    plt.plot(test_preds,  label="Predicted SOC")
    plt.legend()
    plt.title(f"Trial {trial.number} Test SOC Prediction")
    plt.savefig(Path(folder) / "test_plot.png")
    plt.close()

    # Ziel: minimale Validierungs-RMSE
    min_val = min(r["val_rmse"] for r in log_rows)
    return min_val

# Optuna Studienlauf
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Beste Hyperparameter
best_params = study.best_params
print(f"Beste Hyperparameter: {best_params}")

## Train final model mit erneutem HPT-Split (kein Re-Scaling)
model, feature_scaler, log_rows, df_val, df_test = train_online(
    epochs=50, lr=best_params["lr"],
    hidden_size=best_params["hidden_size"], dropout=best_params["dropout"],
    patience=10, out_dir="final_model",
    train_data=train_scaled_glob,
    df_val=df_val_glob, df_test=df_test_glob,
    feature_scaler=feature_scaler_glob
)
# Finale Bewertung ohne Neu-Scaling
val_mse, val_preds, val_labels = evaluate_onechunk_seq2seq(model, df_val, device)
test_mse, test_preds, test_labels = evaluate_onechunk_seq2seq(model, df_test, device)
val_rmse = math.sqrt(val_mse)
test_rmse = math.sqrt(test_mse)
print(f"[INFO] Finale Bewertung -> Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

# Speichern des finalen Modells und Scalers
torch.save(model.state_dict(), "final_model.pth")
with open("feature_scaler.pkl", "wb") as f:
    pickle.dump(feature_scaler, f)

print("Training abgeschlossen und Modelle gespeichert.")