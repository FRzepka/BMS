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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # besser in reinen Terminalscreens

# Gerät auswählen und cuDNN optimieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Output dir für alle Outputs
output_dir = Path("test_standardscaler")
output_dir.mkdir(exist_ok=True)
print(f"Saving all outputs to: {output_dir}")

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

    feats = ["Voltage[V]", "Current[A]"]
    # Trainingsdaten laden und Timestamp
    train_dfs = {}
    for name in train_cells:
        df = cells[name].copy()
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        train_dfs[name] = df

    # Skalar fitten (StandardScaler)
    df_all_train = pd.concat(train_dfs.values(), ignore_index=True)
    scaler = StandardScaler().fit(df_all_train[feats])

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
    i1, i2 = int(L*0.4), int(L*0.8)
    df_val = df3.iloc[:i1].copy()
    df_test = df3.iloc[i2:].copy()
    df_val[feats] = scaler.transform(df_val[feats])
    df_test[feats] = scaler.transform(df_test[feats])

    return train_scaled, df_val, df_test, train_cells, val_cell

# Windowed Dataset für schnelles Training
class WindowedDataset(Dataset):
    def __init__(self, df, seq_len=100):
        self.data = df[["Voltage[V]", "Current[A]"]].values
        self.labels = df["SOC_ZHU"].values
        self.seq_len = seq_len

    def __len__(self):
        # non-overlapping windows
        return (len(self.labels) - self.seq_len) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.from_numpy(self.data[start:start + self.seq_len]).float()
        # return sequence of labels, not just the last one
        y = torch.from_numpy(self.labels[start:start + self.seq_len]).float()
        return x, y

# 1) add weight‐init helper
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
                nn.ReLU(),
                nn.Dropout(dropout),   # nur hier Dropout
                nn.Linear(mlp_hidden, 1),
                nn.Sigmoid()
            )

        def forward(self, x, hidden):
            # out: (batch, seq_len, hidden_size)
            out, hidden = self.lstm(x, hidden)
            batch, seq_len, hid = out.size()
            # apply MLP to every timestep
            out_flat = out.contiguous().view(batch * seq_len, hid)
            soc_flat = self.mlp(out_flat)               # (batch*seq_len, 1)
            soc = soc_flat.view(batch, seq_len)         # (batch, seq_len)
            return soc, hidden
    model = SOCModel().to(device)
    # 2) init weights & optimize cuDNN for multi-layer LSTM
    model.apply(init_weights)
    model.lstm.flatten_parameters()
    return model

# Neuer Helper für Validierung
def evaluate(model, df_val, seq_len, batch_size, device):
    model.eval()
    ds_val = WindowedDataset(df_val, seq_len)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    criterion = nn.MSELoss()
    # hidden init
    h = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size, device=device)
    c = torch.zeros_like(h)
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for x_b, y_b in dl_val:
            x_b, y_b = x_b.to(device), y_b.to(device)
            pred, (h, c) = model(x_b, (h, c))
            loss = criterion(pred, y_b)
            total_loss += loss.item()
            steps += 1
            h, c = h.detach(), c.detach()
    return total_loss / steps if steps>0 else float('inf')

# Training mit Mixed Precision und Batch-Fenstern
def train_online(epochs=20, lr=1e-3, seq_len=100, batch_size=1):  # batch_size=1 for true stateful
    train_scaled, df_val, df_test, train_cells, val_cell = load_data()
    print("Training auf Zellen:", train_cells)

    # Rohdaten-Plots
    for name, df in train_scaled.items():
        plt.figure(figsize=(10,4))
        plt.plot(df['timestamp'], df['SOC_ZHU'], label=name)
        plt.title(f"Train SOC {name}")
        plt.tight_layout(); plt.savefig(output_dir / f"train_{name}_plot.png"); plt.close()
    plt.figure(figsize=(8,4))
    plt.plot(df_val['timestamp'], df_val['SOC_ZHU']); plt.title("Val SOC"); plt.tight_layout(); plt.savefig(output_dir / "val_data_plot.png"); plt.close()
    plt.figure(figsize=(8,4))
    plt.plot(df_test['timestamp'], df_test['SOC_ZHU']); plt.title("Test SOC"); plt.tight_layout(); plt.savefig(output_dir / "test_data_plot.png"); plt.close()

    model = build_model()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler steuert nun am Val-Loss
    scheduler = ReduceLROnPlateau(optim, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    scaler = GradScaler()  # remove deprecated 'device' kwarg
    best_val_loss = float('inf')
    log_rows = []  # für Excel-Log

    for ep in range(1, epochs+1):
        model.train()
        total_loss, total_steps = 0.0, 0
        print(f"\n=== Epoch {ep}/{epochs} ===")
        for name, df in train_scaled.items():
            # Anzahl Fenster entspricht jetzt len(ds)
            ds = WindowedDataset(df, seq_len)
            print(f"--> {name}, Schritte: {len(ds)}")
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,    # kein Shuffle
                drop_last=False,   # keep all windows
                num_workers=4,
                pin_memory=True
            )
            # Hidden States nur einmal pro Zelle initialisieren
            h = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size, device=device)
            c = torch.zeros_like(h)
            for x_b, y_b in tqdm(dl, desc=f"{name} Ep{ep}", leave=True):
                x_b, y_b = x_b.to(device), y_b.to(device)
                optim.zero_grad()
                with autocast():
                    pred, (h, c) = model(x_b, (h, c))
                    loss = criterion(pred, y_b)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optim)
                scaler.update()
                # detach für nächsten Batch
                h, c = h.detach(), c.detach()
                total_loss += loss.item()
                total_steps += 1
        avg_loss = total_loss / total_steps
        print(f"Epoch {ep} Training abgeschlossen, avg train loss={avg_loss:.6f}")

        # --- Validation nach jeder Epoche ---
        val_loss = evaluate(model, df_val, seq_len, batch_size, device)
        print(f"Epoch {ep} Validation abgeschlossen, val loss={val_loss:.6f}")
        scheduler.step(val_loss)
        # Log und Excel aktualisieren
        log_rows.append({"epoch": ep, "train_loss": avg_loss, "val_loss": val_loss})
        pd.DataFrame(log_rows).to_csv(output_dir / "loss_log.csv", index=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_online_soc.pth")
            print(f"  Neuer Best-Checkpoint (val loss={val_loss:.6f}) gespeichert")

# Test: Schritt-für-Schritt Streaming-Inferenz ohne Fensterpuffer
def test_online():
    _, df_val, df_test, train_cells, val_cell = load_data()
    print("Test auf Zelle:", val_cell)
    model = build_model()
    model.load_state_dict(torch.load(output_dir / "best_online_soc.pth", map_location=device))
    model.eval()

    # Streaming-Test auf Validierungsdaten
    print("\n=== Streaming Test on Validation Data ===")
    h_val = torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size, device=device)
    c_val = torch.zeros_like(h_val)
    preds_val, gts_val = [], []
    for v, i, y_true in tqdm(zip(df_val['Voltage[V]'].values,
                                 df_val['Current[A]'].values,
                                 df_val['SOC_ZHU'].values),
                              total=len(df_val), desc="Testing Val"):
        x = torch.tensor([[v, i]], dtype=torch.float32, device=device).view(1,1,2)
        pred, (h_val, c_val) = model(x, (h_val, c_val))
        preds_val.append(pred.item())
        gts_val.append(y_true)
    mae_val = np.mean(np.abs(np.array(preds_val) - np.array(gts_val)))
    rmse_val = np.sqrt(np.mean((np.array(preds_val) - np.array(gts_val))**2))
    print(f"Validation Stream - MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")

    h = torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size, device=device)
    c = torch.zeros_like(h)
    preds, gts = [], []
    ds = df_test
    # TQDM-Balken im Test
    print("\n=== Streaming Test on Test Data ===")
    for v, i in tqdm(zip(ds['Voltage[V]'].values, ds['Current[A]'].values),
                      total=len(ds), desc="Testing"):
        x = torch.tensor([[v, i]], dtype=torch.float32, device=device).view(1,1,2)
        pred, (h, c) = model(x, (h, c))
        preds.append(pred.item())
        gts.append(ds['SOC_ZHU'].iloc[len(preds)-1])
    preds, gts = np.array(preds), np.array(gts)
    mae = np.mean(np.abs(preds - gts)); rmse = np.sqrt(np.mean((preds - gts)**2))
    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # Plots wie gehabt
    plt.figure(figsize=(10,4))
    plt.plot(timestamps, gts, 'k-', label="GT"); plt.plot(timestamps, preds, 'r-')
    plt.title("Online Final Test"); plt.annotate(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}", xy=(0.01,0.95), xycoords='axes fraction', va='top')
    plt.tight_layout(); plt.savefig(output_dir / "final_online_plot.png"); plt.close()
    zoom_n = min(50000, len(preds))
    for name, seg in [("Start", slice(0, zoom_n)), ("End", slice(-zoom_n, None))]:
        plt.figure(figsize=(10,4))
        plt.plot(timestamps[seg], gts[seg], 'k-'); plt.plot(timestamps[seg], preds[seg], 'r-')
        plt.title(f"Zoom {name}"); plt.tight_layout(); plt.savefig(output_dir / f"zoom_{name.lower()}_online_plot.png"); plt.close()

if __name__ == "__main__":
    train_online()
    test_online()
