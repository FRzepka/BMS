import os
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
from sklearn.preprocessing import MinMaxScaler  # Changed to MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv

DEBUG_SMALL = False       # False = Full-Run mit allen Zellen und allen Daten
DEBUG_ROWS = 1000         # maximal Anzahl Zeilen pro Split im Debug-Modus

# Gerät auswählen und cuDNN optimieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Output dir erstellen
output_dir = Path("test_minmax")
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
    print(f"-> load_data start, base_path={base_path}")
    base = Path(base_path)
    cells = load_cell_data(base)
    print(f"   Found {len(cells)} cells: {list(cells.keys())}")
    train_scaled = {}
    all_val, all_test = [], []
    for name, df in cells.items():
        print(f"   Processing cell '{name}' with {len(df)} samples")
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        # neue Features
        df['dt'] = df['timestamp'].diff().dt.total_seconds().fillna(0.0)
        df['coulomb'] = (df['Current[A]'] * df['dt']).cumsum()
        feats = ["Voltage[V]", "Current[A]", "dt", "coulomb"]
        # globales Scaling
        scaler = MinMaxScaler()
        df[feats] = scaler.fit_transform(df[feats])
        L = len(df)
        train_end = int(L * 0.7)
        val_end = int(L * 0.85)
        test_end = L
        train_scaled[name] = df.iloc[:train_end]
        all_val.append(df.iloc[train_end:val_end])
        all_test.append(df.iloc[val_end:test_end])
        print(f"     Splits -> train: {train_end}, val: {val_end-train_end}, test: {test_end-val_end}")
    df_val = pd.concat(all_val)
    df_test = pd.concat(all_test)

    if DEBUG_SMALL:
        first = list(train_scaled.keys())[0]
        train_scaled = { first: train_scaled[first].head(DEBUG_ROWS) }
        df_val = df_val.head(DEBUG_ROWS)
        df_test = df_test.head(DEBUG_ROWS)
        print(f"-> DEBUG_SMALL: using only cell '{first}' and first {DEBUG_ROWS} samples per Split")

    print(f"-> load_data done, train cells={len(train_scaled)}, val samples={len(df_val)}, test samples={len(df_test)}")
    return train_scaled, df_val, df_test, list(train_scaled.keys()), None

# neues Dataset für Seq-to-One mit Sliding Window
class WindowDataset(Dataset):
    def __init__(self, df, window_size=50):
        feats = df.columns.intersection(["Voltage[V]", "Current[A]", "dt", "coulomb"])
        data = df[feats].values
        labels = df["SOC_ZHU"].values
        self.x = [data[i:i + window_size] for i in range(len(data) - window_size)]
        self.y = labels[window_size:]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.tensor(self.x[i], dtype=torch.float32), torch.tensor(self.y[i], dtype=torch.float32)

# Weight-init helper
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

# Modell mit Seq-to-One Option und ohne Sigmoid am Ende
def build_model(input_size=4, hidden_size=32, num_layers=1, dropout=0.0, mlp_hidden=16, seq_to_one=False):
    class SOCModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq_to_one = seq_to_one
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=0.0)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1)  # no Sigmoid here
            )

        def forward(self, x, hidden):
            out, hidden = self.lstm(x, hidden)
            if self.seq_to_one:
                last = out[:, -1, :]  # (B, H)
                raw = self.mlp(last)  # (B,1)
                soc = torch.clamp(raw.squeeze(1), 0.0, 1.0)
                return soc, hidden
            else:
                b, sl, h = out.size()
                flat = out.contiguous().view(b * sl, h)
                soc_flat = self.mlp(flat).view(b, sl)
                soc = torch.clamp(soc_flat, 0.0, 1.0)
                return soc, hidden

    model = SOCModel().to(device)
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

# Online-Validierungsfunktion (Schritt-für-Schritt)
def evaluate_online(model, df, device, desc="Validating"):
    """Online-Validierung: Schritt-für-Schritt wie im echten Betrieb."""
    model.eval()
    h, c = init_hidden(model, device=device)
    preds, gts = [], []
    feats = ["Voltage[V]", "Current[A]", "dt", "coulomb"]
    with torch.no_grad():
        for idx in tqdm(range(len(df)), total=len(df), desc=desc, leave=True):
            row = df.iloc[idx]
            # Werte als Float-Array umwandeln, kein object dtype
            vals = row[feats].astype(float).values
            x = torch.tensor(vals, dtype=torch.float32, device=device).view(1,1,len(feats))
            pred, (h, c) = model(x, (h, c))
            preds.append(pred.item())
            gts.append(row["SOC_ZHU"])
    preds, gts = np.array(preds), np.array(gts)
    mse = np.mean((preds - gts) ** 2)
    return mse, preds, gts

# Helper: downsample DataFrame to at most max_points
def subsample(df, max_points=10000):
    n = len(df)
    if n <= max_points:
        return df
    step = max(1, n // max_points)
    return df.iloc[::step].reset_index(drop=True)

# Training-Funktion
def train_online(epochs=5, lr=1e-3, batch_size=128, online_train=False, log_csv_path="training_log.csv"):
    print(f"-> train_online start: epochs={epochs}, lr={lr}, batch_size={batch_size}, online_train={online_train}")
    train_scaled, df_val, df_test, train_cells, val_cell = load_data()
    print(f"   train_cells={train_cells}, df_val.shape={df_val.shape}, df_test.shape={df_test.shape}")

    # Save log to output directory
    log_csv_path = output_dir / log_csv_path

    # Rohdaten-Plots mit Voltage und Current
    for name, df in train_scaled.items():
        # SOC Plot
        dfp = subsample(df)                                      # <--- subsample
        plt.figure(figsize=(12,8))
        plt.subplot(3, 1, 1)
        plt.plot(dfp['timestamp'], dfp['SOC_ZHU'], 'b-', label="SOC")
        plt.title(f"Training Data - {name} - {len(df)} samples")
        plt.ylabel("SOC")
        plt.legend()
        
        # Voltage Plot
        plt.subplot(3, 1, 2)
        plt.plot(dfp['timestamp'], dfp['Voltage[V]'], 'g-', label="Voltage [scaled]")
        plt.ylabel("Voltage (scaled)")
        plt.legend()
        
        # Current Plot
        plt.subplot(3, 1, 3)
        plt.plot(dfp['timestamp'], dfp['Current[A]'], 'r-', label="Current [scaled]")
        plt.xlabel("Zeit")
        plt.ylabel("Current (scaled)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f"train_data_plot.png")
        plt.close()
    
    # Validation data plot
    dfp = subsample(df_val)
    plt.figure(figsize=(12,8))
    plt.subplot(3, 1, 1)
    plt.plot(dfp['timestamp'], dfp['SOC_ZHU'], 'b-', label="SOC")
    plt.title(f"Validation Data - {len(df_val)} samples")
    plt.ylabel("SOC")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(dfp['timestamp'], dfp['Voltage[V]'], 'g-', label="Voltage [scaled]")
    plt.ylabel("Voltage (scaled)")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(dfp['timestamp'], dfp['Current[A]'], 'r-', label="Current [scaled]")
    plt.xlabel("Zeit")
    plt.ylabel("Current (scaled)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "val_data_plot.png")
    plt.close()
    
    # Test data plot
    dfp = subsample(df_test)
    plt.figure(figsize=(12,8))
    plt.subplot(3, 1, 1)
    plt.plot(dfp['timestamp'], dfp['SOC_ZHU'], 'b-', label="SOC")
    plt.title(f"Test Data - {len(df_test)} samples")
    plt.ylabel("SOC")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(dfp['timestamp'], dfp['Voltage[V]'], 'g-', label="Voltage [scaled]")
    plt.ylabel("Voltage (scaled)")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(dfp['timestamp'], dfp['Current[A]'], 'r-', label="Current [scaled]")
    plt.xlabel("Zeit")
    plt.ylabel("Current (scaled)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "test_data_plot.png")
    plt.close()

    # für Online-Train (Stateful-Step) seq_to_one=False, sonst Sliding-Window Seq-to-One
    model = build_model(input_size=4, hidden_size=32, num_layers=1,
                        dropout=0.0, mlp_hidden=16,
                        seq_to_one=not online_train)
    print(f"   Model built: {model}")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optim, mode='min', patience=3, factor=0.5)
    criterion = nn.HuberLoss(delta=0.1)
    scaler = GradScaler('cuda')
    best_val_loss = float('inf')

    # Logging vorbereiten
    log_fields = ["epoch", "train_loss", "val_loss"]
    log_rows = []

    # Fortschrittsbalken über Epochen
    for ep in tqdm(range(1, epochs+1), desc="Epochs", leave=True):
        print(f"\n--- Epoch {ep}/{epochs} start")
        model.train()
        total_loss, total_steps = 0.0, 0
        print(f"\n=== Epoch {ep}/{epochs} ===")
        # Fortschrittsbalken über Zellen in dieser Epoche
        for name in tqdm(train_scaled.keys(), desc=f"Cells Ep{ep}", leave=False):
            df = train_scaled[name]
            if not online_train:
                # Batch training
                ds = WindowDataset(df, window_size=50)
                print(f"--> {name}, Batches: {len(ds)}")
                dl = DataLoader(
                    ds,
                    batch_size=32,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                # bereits vorhandener tqdm pro Batch:
                for x_b, y_b in tqdm(dl, desc=f"{name} Ep{ep} train", leave=False):
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    h, c = init_hidden(model, batch_size=x_b.size(0), device=device)
                    optim.zero_grad()
                    with autocast('cuda'):
                        pred, _ = model(x_b, (h, c))
                        loss = criterion(pred, y_b)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optim)
                    scaler.update()
                    total_loss += loss.item()
                    total_steps += 1
            else:
                # Online-Training: Schritt-für-Schritt
                print(f"--> {name} (Online-Train, {len(df)} steps)")
                # initialize hidden state once per sequence
                h, c = init_hidden(model, batch_size=1, device=device)
                feats = ["Voltage[V]", "Current[A]", "dt", "coulomb"]
                for idx in tqdm(range(len(df)), total=len(df), desc=f"{name} Ep{ep}", leave=True):
                    row = df.iloc[idx]
                    vals = row[feats].astype(float).values
                    x = torch.tensor(vals, dtype=torch.float32, device=device).view(1,1,len(feats))
                    y_true = float(row["SOC_ZHU"])
                    y = torch.tensor([[y_true]], dtype=torch.float32, device=device)
                    optim.zero_grad()
                    with autocast('cuda'):
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

        avg_loss = total_loss / total_steps
        print(f"--- Epoch {ep} train done, avg_loss={avg_loss:.6f}")

        # Online-Validierung nach jeder Epoche
        # tqdm für Validierung schon in evaluate_online enthalten
        val_loss, val_preds, val_gts = evaluate_online(model, df_val, device, desc=f"Validating Epoch {ep}")
        print(f"--- Epoch {ep} validation loss={val_loss:.6f}")
        scheduler.step(val_loss)
        
        # Validation Plot für diese Epoche erstellen
        plt.figure(figsize=(12,8))
        plt.subplot(3, 1, 1)
        plt.plot(df_val['timestamp'].values, val_gts, 'k-', label="Ground Truth")
        plt.plot(df_val['timestamp'].values, val_preds, 'r-', label="Prediction")
        plt.title(f"Validation - Epoch {ep}")
        plt.ylabel("SOC")
        plt.legend()
        plt.annotate(f"Val Loss: {val_loss:.6f}", xy=(0.01,0.95), xycoords='axes fraction', va='top')
        
        # Add voltage and current plots
        plt.subplot(3, 1, 2)
        plt.plot(df_val['timestamp'].values, df_val['Voltage[V]'].values, 'g-', label="Voltage [scaled]")
        plt.ylabel("Voltage (scaled)")
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(df_val['timestamp'].values, df_val['Current[A]'].values, 'b-', label="Current [scaled]")
        plt.xlabel("Zeit")
        plt.ylabel("Current (scaled)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f"validation_epoch_{ep}.png")
        plt.close()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_online_soc.pth")
            print(f"  Neuer Best-Checkpoint (val loss={val_loss:.6f}) gespeichert")

        # Logging: Zeile anhängen
        log_rows.append({"epoch": ep, "train_loss": avg_loss, "val_loss": val_loss})

        # Logging: CSV nach jeder Epoche aktualisieren
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
            writer.writerows(log_rows)

# Test-Funktion
def test_online(log_csv_path="training_log.csv"):
    train_scaled, df_val, df_test, train_cells, val_cell = load_data()
    print("Test auf Zelle:", val_cell)
    model = build_model(input_size=4, hidden_size=32, num_layers=1, dropout=0.0, mlp_hidden=16, seq_to_one=True)
    model.load_state_dict(torch.load(output_dir / "best_online_soc.pth", map_location=device))
    model.eval()
    
    # Update log path to use output directory
    log_csv_path = output_dir / log_csv_path
    
    # Test auf den Validierungsdaten
    print("\n=== Test on Validation Data ===")
    # tqdm für Test ebenfalls in evaluate_online
    val_loss, preds_val, gts_val = evaluate_online(model, df_val, device, desc="Testing on Val Data")
    timestamps_val = df_val['timestamp'].values
    mae_val = np.mean(np.abs(preds_val - gts_val))
    rmse_val = np.sqrt(np.mean((preds_val - gts_val)**2))
    print(f"Validation Data - MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")
    
    # Plot für Validierungsdaten mit Voltage und Current
    plt.figure(figsize=(12,8))
    plt.subplot(3, 1, 1)
    plt.plot(timestamps_val, gts_val, 'k-', label="Ground Truth")
    plt.plot(timestamps_val, preds_val, 'r-', label="Prediction")
    plt.title(f"Final Test on Validation Data ({len(df_val)} samples)")
    plt.ylabel("SOC")
    plt.legend()
    plt.annotate(f"MAE: {mae_val:.4f}\nRMSE: {rmse_val:.4f}", xy=(0.01,0.95), xycoords='axes fraction', va='top')
    
    plt.subplot(3, 1, 2)
    plt.plot(timestamps_val, df_val['Voltage[V]'].values, 'g-', label="Voltage [scaled]")
    plt.ylabel("Voltage (scaled)")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(timestamps_val, df_val['Current[A]'].values, 'b-', label="Current [scaled]")
    plt.xlabel("Zeit")
    plt.ylabel("Current (scaled)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "final_val_test_plot.png")
    plt.close()
    
    # Test auf den Testdaten
    print("\n=== Test on Test Data ===")
    test_loss, preds, gts = evaluate_online(model, df_test, device, desc="Testing on Test Data")
    timestamps = df_test['timestamp'].values
    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts)**2))
    print(f"Test Data - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # Logging: Testresultate an CSV anhängen
    try:
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["dataset", "mae", "rmse"])
            writer.writerow(["validation", mae_val, rmse_val])
            writer.writerow(["test", mae, rmse])
    except Exception as e:
        print(f"Fehler beim Schreiben der Testresultate in die CSV: {e}")

    # Plots für Testdaten mit Voltage und Current
    plt.figure(figsize=(12,8))
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, gts, 'k-', label="Ground Truth")
    plt.plot(timestamps, preds, 'r-', label="Prediction")
    plt.title(f"Final Test on Test Data ({len(df_test)} samples)")
    plt.ylabel("SOC")
    plt.legend()
    plt.annotate(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}", xy=(0.01,0.95), xycoords='axes fraction', va='top')
    
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, df_test['Voltage[V]'].values, 'g-', label="Voltage [scaled]")
    plt.ylabel("Voltage (scaled)")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(timestamps, df_test['Current[A]'].values, 'b-', label="Current (scaled)")
    plt.xlabel("Zeit")
    plt.ylabel("Current (scaled)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "final_test_plot.png")
    plt.close()
    
    zoom_n = min(5000, len(preds))
    for name, seg in [("Start", slice(0, zoom_n)), ("End", slice(-zoom_n, None))]:
        plt.figure(figsize=(12,8))
        
        plt.subplot(3, 1, 1)
        plt.plot(timestamps[seg], gts[seg], 'k-', label="Ground Truth")
        plt.plot(timestamps[seg], preds[seg], 'r-', label="Prediction")
        plt.title(f"Zoom {name} - Test Data")
        plt.ylabel("SOC")
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(timestamps[seg], df_test['Voltage[V]'].values[seg], 'g-', label="Voltage [scaled]")
        plt.ylabel("Voltage (scaled)")
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(timestamps[seg], df_test['Current[A]'].values[seg], 'b-', label="Current (scaled)")
        plt.xlabel("Zeit")
        plt.ylabel("Current (scaled)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f"zoom_{name.lower()}_test_plot.png")
        plt.close()

if __name__ == "__main__":
    # HIER BATCHSIZE ÄNDERN - nur ein Ort für die Änderung
    BATCH_SIZE = 128  # <<< HIER DIE BATCH-GRÖßE EINSTELLEN
    
    # Full-Run: längere Laufzeit, echtes stateful-Online-Training
    train_online(epochs=20, lr=1e-4, batch_size=BATCH_SIZE,
                 online_train=True, log_csv_path="training_log.csv")
    test_online(log_csv_path="training_log.csv")
