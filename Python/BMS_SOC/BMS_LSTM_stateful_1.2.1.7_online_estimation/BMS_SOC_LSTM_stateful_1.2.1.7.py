import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # besser in reinen Terminalscreens
import traceback  # new for stack traces

# Gerät auswählen und cuDNN optimieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Datenlade-Funktion
def load_cell_data(data_dir: Path):
    print(f"DEBUG load_cell_data: scanning directory {data_dir}")
    dataframes = {}
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("MGFarm_18650_"):
            print(f"DEBUG load_cell_data: entering folder {folder.name}")
            dfp = folder / "df.parquet"
            if dfp.exists():
                df = pd.read_parquet(dfp)
                print(f"DEBUG load_cell_data: loaded {folder.name} with {len(df)} rows")
                dataframes[folder.name] = df
            else:
                print(f"Warning: {dfp} fehlt")
    print(f"DEBUG load_cell_data: returning {len(dataframes)} dataframes")
    return dataframes

# Daten vorbereiten
def load_data(base_path: str = "/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes"):
    base = Path(base_path)
    cells = load_cell_data(base)
    names = sorted(cells.keys())
    train_cells, val_cell = names[:2], names[2]

    # 1) Rohdaten einlesen und Timestamp hinzufügen
    train_raw = {}
    for name in train_cells:
        df = cells[name].copy()
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        train_raw[name] = df

    df3 = cells[val_cell].copy()
    df3['timestamp'] = pd.to_datetime(df3['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    L = len(df3)
    i1, i2 = int(L*0.4), int(L*0.8)
    df_val_raw = df3.iloc[:i1].copy()
    df_test_raw = df3.iloc[i2:].copy()

    # 2) Zwei MinMaxScaler wie in testplots
    feats_v = ["Voltage[V]", "SOC_ZHU"]
    feats_c = ["Current[A]"]
    mm_v = MinMaxScaler(feature_range=(0,1)).fit(
        pd.concat([train_raw[n][feats_v] for n in train_cells], ignore_index=True)
    )
    mm_c = MinMaxScaler(feature_range=(-1,1)).fit(
        pd.concat([train_raw[n][feats_c] for n in train_cells], ignore_index=True)
    )

    # 3) Daten skalieren
    train_scaled = {}
    for n in train_cells:
        df = train_raw[n].copy()
        df[feats_v] = mm_v.transform(df[feats_v])
        df[feats_c] = mm_c.transform(df[feats_c])
        train_scaled[n] = df

    df_val = df_val_raw.copy()
    df_val[feats_v] = mm_v.transform(df_val_raw[feats_v])
    df_val[feats_c] = mm_c.transform(df_val_raw[feats_c])

    df_test = df_test_raw.copy()
    df_test[feats_v] = mm_v.transform(df_test_raw[feats_v])
    df_test[feats_c] = mm_c.transform(df_test_raw[feats_c])

    return train_scaled, df_val, df_test, train_cells, val_cell

# Windowed Dataset für schnelles Training
class WindowedDataset(Dataset):
    def __init__(self, df, seq_len=100):  # vorher 50
        self.data = df[["Voltage[V]", "Current[A]"]].values
        self.labels = df["SOC_ZHU"].values
        self.seq_len = seq_len

    def __len__(self):
        return len(self.labels) - self.seq_len

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx+self.seq_len]).float()
        y = torch.tensor(self.labels[idx+self.seq_len], dtype=torch.float32)
        return x, y

# Modell: LSTM + Dropout + MLP-Head
def build_model(input_size=2, hidden_size=32, num_layers=1, dropout=0.2, mlp_hidden=16):  # angepasst
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
                nn.Sigmoid(),
                nn.Dropout(dropout),   # nur hier Dropout
                nn.Linear(mlp_hidden, 1)
            )

        def forward(self, x, hidden):
            out, hidden = self.lstm(x, hidden)
            last = out[:, -1, :]
            soc = self.mlp(last)
            return soc.squeeze(-1), hidden
    return SOCModel().to(device)

# Training mit Mixed Precision und Batch-Fenstern
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_online(epochs=20, lr=1e-3, seq_len=100, batch_size=128):  # vorher 64
    print("train_online: loading data...")
    train_scaled, df_val, df_test, train_cells, val_cell = load_data()
    print(f"train_online: Training on cells {train_cells}, validating {val_cell}")

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
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2)
    criterion = nn.MSELoss()
    scaler = GradScaler()  # remove deprecated 'device' kwarg
    best_val_loss = float('inf')

    for ep in range(1, epochs+1):
        print(f"\n=== Epoch {ep}/{epochs} START ===")  # more verbose
        model.train()
        total_loss, total_steps = 0.0, 0
        for name, df in train_scaled.items():
            print(f"  Epoch {ep}: preparing DataLoader for {name}, seq_len={seq_len}, batch_size={batch_size}")
            ds = WindowedDataset(df, seq_len)
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,        # erhöht für schnellere Datenvorbereitung
                pin_memory=True       # beschleunigt Übertragung auf GPU
            )
            print(f"  Epoch {ep}: start training loop for {name}, steps ~ {len(dl)}")
            # TQDM-Balken pro Batch
            for step, (x_b, y_b) in enumerate(tqdm(dl, desc=f"{name} Ep{ep}", leave=True), 1):
                if step % 500 == 0:
                    print(f"    {name} Ep{ep}: at batch {step}/{len(dl)}")
                x_b, y_b = x_b.to(device), y_b.to(device)
                h0 = torch.zeros(model.lstm.num_layers, x_b.size(0), model.lstm.hidden_size, device=device)
                c0 = torch.zeros_like(h0)
                optim.zero_grad()
                with autocast():
                    pred, _ = model(x_b, (h0, c0))
                    loss = criterion(pred, y_b)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optim)
                scaler.update()
                total_loss += loss.item()
                total_steps += 1

        print(f"=== Epoch {ep} TRAINING DONE, starting validation ===")
        # --- Validation nach jeder Epoche ---
        model.eval()
        val_ds = WindowedDataset(df_val, seq_len)
        val_dl = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, drop_last=True,
                            num_workers=4, pin_memory=True)
        h_val = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size, device=device)
        c_val = torch.zeros_like(h_val)
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for x_v, y_v in tqdm(val_dl, desc=f"Validation Ep{ep}", leave=False):
                x_v, y_v = x_v.to(device), y_v.to(device)
                pred_v, (h_val, c_val) = model(x_v, (h_val, c_val))
                loss_v = criterion(pred_v, y_v)
                val_loss += loss_v.item()
                val_steps += 1
                h_val, c_val = h_val.detach(), c_val.detach()
        avg_val = val_loss / val_steps
        print(f"Epoch {ep} completed, validation loss={avg_val:.6f}")

        scheduler.step(avg_val)
        if avg_val < best_val_loss:
            print(f"  New best model at epoch {ep}, saving checkpoint")
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_online_soc.pth")

# Test: Schritt-für-Schritt Streaming-Inferenz ohne Fensterpuffer
def test_online():
    print("test_online: loading data and model...")
    _, df_val, df_test, train_cells, val_cell = load_data()
    print("Test auf Zelle:", val_cell)
    model = build_model()
    print("test_online: loading state dict from 'best_online_soc.pth'")
    model.load_state_dict(torch.load("best_online_soc.pth", map_location=device))
    model.eval()
    print(f"test_online: starting streaming inference on {val_cell}, total points={len(df_test)}")

    h = torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size, device=device)
    c = torch.zeros_like(h)
    preds, gts = [], []
    ds = df_test
    # TQDM-Balken im Test
    for idx, (v, i) in enumerate(zip(ds['Voltage[V]'].values, ds['Current[A]'].values), 1):
        if idx % 10000 == 0:
            print(f"  test_online: reached step {idx}/{len(ds)}")
        x = torch.tensor([[v, i]], dtype=torch.float32, device=device).view(1,1,2)
        pred, (h, c) = model(x, (h, c))
        preds.append(pred.item())
        gts.append(ds['SOC_ZHU'].iloc[len(preds)-1])
    print(f"test_online: finished inference, generated {len(preds)} preds")
    preds, gts = np.array(preds), np.array(gts)
    timestamps = df_test['timestamp'].values
    mae = np.mean(np.abs(preds - gts)); rmse = np.sqrt(np.mean((preds - gts)**2))
    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # Plots wie gehabt
    plt.figure(figsize=(10,4))
    plt.plot(timestamps, gts, 'k-', label="GT"); plt.plot(timestamps, preds, 'r-')
    plt.title("Online Final Test"); plt.annotate(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}", xy=(0.01,0.95), xycoords='axes fraction', va='top')
    plt.tight_layout(); plt.savefig("final_online_plot.png"); plt.close()
    zoom_n = min(50000, len(preds))
    for name, seg in [("Start", slice(0, zoom_n)), ("End", slice(-zoom_n, None))]:
        plt.figure(figsize=(10,4))
        plt.plot(timestamps[seg], gts[seg], 'k-'); plt.plot(timestamps[seg], preds[seg], 'r-')
        plt.title(f"Zoom {name}"); plt.tight_layout(); plt.savefig(f"zoom_{name.lower()}_online_plot.png"); plt.close()

if __name__ == "__main__":
    try:
        print("Main: START")
        train_online()
        print("Main: train_online() finished")
        test_online()
        print("Main: DONE")
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
