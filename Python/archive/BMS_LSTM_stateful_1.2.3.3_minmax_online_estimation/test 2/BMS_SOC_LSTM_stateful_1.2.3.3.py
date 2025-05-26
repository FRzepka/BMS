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

# Konstanten
BATCH_SIZE = 5000    # <<< Hier die Batch-Größe einstellen

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
    names = sorted(cells.keys())
    train_cells, val_cell = names[:2], names[2]

    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU"]  # jetzt auch SOH skalieren
    # Trainingsdaten laden und Timestamp
    train_dfs = {}
    for name in train_cells:
        df = cells[name].copy()
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        train_dfs[name] = df

    # Skalar fitten (Min-Max zwischen 0 und 1)
    df_all_train = pd.concat(train_dfs.values(), ignore_index=True)
    scaler = MaxAbsScaler().fit(df_all_train[feats])

    # Skalierte Trainingsdaten
    train_scaled = {}
    for name, df in train_dfs.items():
        df2 = df.copy()
        df2[feats] = scaler.transform(df2[feats])
        train_scaled[name] = df2

    # Validierung/Test der dritten Zelle
    df3 = cells[val_cell].copy()
    df3['timestamp'] = pd.to_datetime(df3['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    L = len(df3)
    i1, i2 = int(L*0.2), int(L*0.4)
    df_val = df3.iloc[:i1].copy()
    df_test = df3.iloc[i1:i2].copy()
    df_val[feats] = scaler.transform(df_val[feats])
    df_test[feats] = scaler.transform(df_test[feats])

    return train_scaled, df_val, df_test, train_cells, val_cell

# Angepasstes Dataset für ganze Zellen
class CellDataset(Dataset):
    def __init__(self, df, batch_size=1024):
        """Dataset für eine ganze Zelle, aufgeteilt in Batches für effizientes Training
        
        Args:
            df: DataFrame mit den Zelldaten
            batch_size: Größe der Batches innerhalb der Zelle
        """
        self.data = df[["Voltage[V]", "Current[A]", "SOH_ZHU"]].values
        self.labels = df["SOC_ZHU"].values
        self.batch_size = batch_size
        
        # Berechne wie viele volle Batches wir haben
        self.n_batches = len(self.data) // batch_size
        if self.n_batches == 0:
            self.n_batches = 1  # Mindestens ein Batch
            self.batch_size = len(self.data)
    
    def __len__(self):
        return self.n_batches  # Anzahl der Batches
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.data))
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
def build_model(input_size=3, hidden_size=64, num_layers=1, dropout=0.2, mlp_hidden=16):
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
    seq = df[["Voltage[V]", "Current[A]", "SOH_ZHU"]].values  # + SOH
    labels = df["SOC_ZHU"].values
    total = len(seq)
    n_chunks = math.ceil(total / BATCH_SIZE)
    h, c = init_hidden(model, batch_size=1, device=device)
    preds = []

    pbar = tqdm(total=n_chunks, desc="Seq2Seq Val", leave=False)
    with torch.no_grad():
        for i in range(n_chunks):
            s = i * BATCH_SIZE
            e = min(s + BATCH_SIZE, total)
            chunk = torch.tensor(seq[s:e], dtype=torch.float32, device=device).unsqueeze(0).contiguous()  # Shape now (1, N, 3)
            model.lstm.flatten_parameters()
            out, (h, c) = model(chunk, (h, c))
            preds.extend(out.squeeze(0).cpu().numpy())
            pbar.update(1)
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
        for idx, (v, i, soh) in enumerate(
            tqdm(zip(df['Voltage[V]'].values,
                     df['Current[A]'].values,
                     df['SOH_ZHU'].values),
                 total=len(df), desc="Validation", leave=False)
        ):
            x = torch.tensor([[v, i, soh]], dtype=torch.float32, device=device).view(1,1,3).contiguous()
            pred, (h, c) = model(x, (h, c))
            preds.append(pred.item())
            gts.append(df['SOC_ZHU'].iloc[idx])
    preds, gts = np.array(preds), np.array(gts)
    return np.mean((preds - gts)**2)

# Training Funktion mit Batch-Training und Seq2Seq-Validierung
def train_online(epochs=30, lr=1e-4, online_train=False, log_csv_path="training_log.csv"):
    train_scaled, df_val, df_test, train_cells, val_cell = load_data()
    print("Training auf Zellen:", train_cells)

    # Rohdaten-Plots
    for name, df in train_scaled.items():
        plt.figure(figsize=(10,4))
        plt.plot(df['timestamp'], df['SOC_ZHU'], label=name)
        plt.title(f"Train SOC {name}")
        plt.tight_layout(); plt.savefig(f"train_{name}_plot.png"); plt.close()
    plt.figure(figsize=(8,4))
    plt.plot(df_val['timestamp'], df_val['SOC_ZHU']); plt.title("Val SOC"); plt.tight_layout(); plt.savefig("val_data_plot.png"); plt.close()
    plt.figure(figsize=(8,4))
    plt.plot(df_test['timestamp'], df_test['SOC_ZHU']); plt.title("Test SOC"); plt.tight_layout(); plt.savefig("test_data_plot.png"); plt.close()

    model = build_model()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optim, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    best_val_loss = float('inf')

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
                ds = CellDataset(df, BATCH_SIZE)
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
                    tqdm(zip(df['Voltage[V]'].values, df['Current[A]'].values, df['SOH_ZHU'].values, df['SOC_ZHU'].values),
                         total=len(df), desc=f"{name} Ep{ep}", leave=True)):
                    x = torch.tensor([[v, i, soh]], dtype=torch.float32, device=device).view(1,1,3).contiguous()
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

        # --- Seq2Seq-Validierung: ein einziger Forward-Pass ---
        print("Seq2Seq-Validation (ein Forward-Pass)...")
        val_mse = evaluate_seq2seq(model, df_val, device)

        val_rmse = math.sqrt(val_mse)
        print(f"Epoch {ep} Validierung abgeschlossen, val RMSE={val_rmse:.6f}")
        scheduler.step(val_mse)
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(model.state_dict(), "best_seq2seq_soc.pth")
            print(f"  Neuer Best-Checkpoint (val RMSE={val_rmse:.6f}) gespeichert")

        # --- Logging: Zeile anhängen ---
        log_rows.append({"epoch": ep, "train_rmse": train_rmse, "val_rmse": val_rmse})

        # --- Logging: CSV nach jeder Epoche aktualisieren ---
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
            writer.writerows(log_rows)

# Test: Seq-to-Seq-Inferenz mit einem Forward-Pass
def test_seq2seq(log_csv_path="training_log.csv"):
    _, _, df_test, _, val_cell = load_data()
    print("Test auf Zelle:", val_cell)
    
    model = build_model()
    model.load_state_dict(torch.load("best_seq2seq_soc.pth", map_location=device))
    model.eval()

    # --- Seq2Seq-Test ---
    test_mse = evaluate_seq2seq(model, df_test, device)
    test_rmse = math.sqrt(test_mse)

    # Get predictions for plotting and MAE calculation
    model.lstm.flatten_parameters()
    feats = torch.tensor(
        df_test[["Voltage[V]", "Current[A]", "SOH_ZHU"]].values,
        dtype=torch.float32, device=device
    ).unsqueeze(0).contiguous()  # Shape now (1, L, 3)
    h0, c0 = init_hidden(model, batch_size=1, device=device)
    h0, c0 = h0.contiguous(), c0.contiguous()

    with torch.backends.cudnn.flags(enabled=False), torch.no_grad():
        model.lstm.flatten_parameters()
        preds, _ = model(feats.contiguous(), (h0, c0))
    preds = preds.squeeze(0).cpu().numpy()
    gts = df_test["SOC_ZHU"].values
    timestamps = df_test['timestamp'].values
    
    test_mae = np.mean(np.abs(preds - gts))
    print(f"Seq2Seq Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

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

if __name__ == "__main__":
    train_online(epochs=500, lr=1e-3, online_train=False, log_csv_path="training_log_seq2seq.csv")
    test_seq2seq(log_csv_path="training_log_seq2seq.csv")