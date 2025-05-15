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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv

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
    
    # Nur die erste Zelle verwenden
    first_cell = names[0]
    print(f"Using only cell: {first_cell}")

    feats = ["Voltage[V]", "Current[A]"]
    # Zellendaten laden und Timestamp hinzufügen
    df = cells[first_cell].copy()
    df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    
    # Gesamtlänge ausgeben
    print(f"Total dataframe length: {len(df)} samples")
    
    # Datenaufteilung: 10% Training, 5% Validierung, 5% Test
    L = len(df)
    train_end = int(L * 0.05)
    val_end = train_end + int(L * 0.01)
    test_end = val_end + int(L * 0.01)
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:test_end].copy()
    
    print(f"Data split for cell {first_cell}:")
    print(f"- Train: {len(df_train)} samples ({len(df_train)/L*100:.1f}%)")
    print(f"- Val: {len(df_val)} samples ({len(df_val)/L*100:.1f}%)")
    print(f"- Test: {len(df_test)} samples ({len(df_test)/L*100:.1f}%)")
    
    # StandardScaler fitten auf den Trainingsdaten
    scaler = StandardScaler().fit(df_train[feats])
    
    # Daten skalieren
    df_train[feats] = scaler.transform(df_train[feats])
    df_val[feats] = scaler.transform(df_val[feats])
    df_test[feats] = scaler.transform(df_test[feats])
    
    # Für das Trainingsformat
    train_scaled = {first_cell: df_train}
    
    return train_scaled, df_val, df_test, [first_cell], first_cell

# Angepasstes Dataset für ganze Zellen
class CellDataset(Dataset):
    def __init__(self, df, batch_size):
        """Dataset für eine ganze Zelle, aufgeteilt in Batches für effizientes Training"""
        self.data = df[["Voltage[V]", "Current[A]"]].values
        self.labels = df["SOC_ZHU"].values
        self.batch_size = batch_size
        
        # Berechne wie viele volle Batches wir haben
        self.n_batches = len(self.data) // batch_size
        if self.n_batches == 0:
            self.n_batches = 1
            self.batch_size = len(self.data)
    
    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.data))
        x = torch.from_numpy(self.data[start:end]).float()
        y = torch.from_numpy(self.labels[start:end]).float()
        return x, y

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

# Modell: LSTM + Dropout + MLP-Head
def build_model(input_size=2, hidden_size=32, num_layers=1, dropout=0.2, mlp_hidden=16):
    class SOCModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=0.0)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1),
                nn.Sigmoid()
            )

        def forward(self, x, hidden):
            out, hidden = self.lstm(x, hidden)
            batch, seq_len, hid = out.size()
            out_flat = out.contiguous().view(batch * seq_len, hid)
            soc_flat = self.mlp(out_flat)
            soc = soc_flat.view(batch, seq_len)
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
    with torch.no_grad():
        # Add tqdm progress bar
        for idx, (v, i) in enumerate(tqdm(zip(df['Voltage[V]'].values, df['Current[A]'].values), 
                                          total=len(df), desc=desc, leave=True)):
            x = torch.tensor([[v, i]], dtype=torch.float32, device=device).view(1,1,2)
            pred, (h, c) = model(x, (h, c))
            preds.append(pred.item())
            gts.append(df['SOC_ZHU'].iloc[idx])
    preds, gts = np.array(preds), np.array(gts)
    mse = np.mean((preds - gts) ** 2)
    return mse, preds, gts

# Training-Funktion
def train_online(epochs=5, lr=1e-3, batch_size=128, online_train=False, log_csv_path="training_log.csv"):
    train_scaled, df_val, df_test, train_cells, val_cell = load_data()
    print("Training auf Zellen:", train_cells)

    # Rohdaten-Plots
    for name, df in train_scaled.items():
        plt.figure(figsize=(10,4))
        plt.plot(df['timestamp'], df['SOC_ZHU'], label="SOC")
        plt.title(f"Training Data - {name} - {len(df)} samples")
        plt.xlabel("Zeit")
        plt.ylabel("SOC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"train_data_plot.png")
        plt.close()
    
    plt.figure(figsize=(10,4))
    plt.plot(df_val['timestamp'], df_val['SOC_ZHU'], label="SOC")
    plt.title(f"Validation Data - {len(df_val)} samples") 
    plt.xlabel("Zeit")
    plt.ylabel("SOC")
    plt.legend()
    plt.tight_layout()
    plt.savefig("val_data_plot.png")
    plt.close()
    
    plt.figure(figsize=(10,4))
    plt.plot(df_test['timestamp'], df_test['SOC_ZHU'], label="SOC")
    plt.title(f"Test Data - {len(df_test)} samples")
    plt.xlabel("Zeit") 
    plt.ylabel("SOC")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_data_plot.png")
    plt.close()

    model = build_model()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optim, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')
    best_val_loss = float('inf')

    # Logging vorbereiten
    log_fields = ["epoch", "train_loss", "val_loss"]
    log_rows = []

    for ep in range(1, epochs+1):
        model.train()
        total_loss, total_steps = 0.0, 0
        print(f"\n=== Epoch {ep}/{epochs} ===")
        for name, df in train_scaled.items():
            if not online_train:
                # Batch training
                ds = CellDataset(df, batch_size)
                print(f"--> {name}, Batches: {len(ds)}")
                dl = DataLoader(
                    ds,
                    batch_size=1,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
                h, c = init_hidden(model, device=device)
                for x_b, y_b in tqdm(dl, desc=f"{name} Ep{ep}", leave=True):
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    optim.zero_grad()
                    with autocast('cuda'):
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
                # Online-Training: Schritt-für-Schritt
                print(f"--> {name} (Online-Train, {len(df)} steps)")
                h, c = init_hidden(model, batch_size=1, device=device)
                for idx, (v, i, y_true) in enumerate(
                    tqdm(zip(df['Voltage[V]'].values, df['Current[A]'].values, df['SOC_ZHU'].values),
                         total=len(df), desc=f"{name} Ep{ep}", leave=True)):
                    x = torch.tensor([[v, i]], dtype=torch.float32, device=device).view(1,1,2)
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
        print(f"Epoch {ep} Training abgeschlossen, avg train loss={avg_loss:.6f}")

        # Online-Validierung nach jeder Epoche
        val_loss, val_preds, val_gts = evaluate_online(model, df_val, device, desc=f"Validating Epoch {ep}")
        print(f"Epoch {ep} Online-Validation abgeschlossen, val loss={val_loss:.6f}")
        scheduler.step(val_loss)
        
        # Validation Plot für diese Epoche erstellen
        plt.figure(figsize=(10,4))
        plt.plot(df_val['timestamp'].values, val_gts, 'k-', label="Ground Truth")
        plt.plot(df_val['timestamp'].values, val_preds, 'r-', label="Prediction")
        plt.title(f"Validation - Epoch {ep}")
        plt.xlabel("Zeit")
        plt.ylabel("SOC")
        plt.legend()
        plt.annotate(f"Val Loss: {val_loss:.6f}", xy=(0.01,0.95), xycoords='axes fraction', va='top')
        plt.tight_layout()
        plt.savefig(f"validation_epoch_{ep}.png")
        plt.close()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_online_soc.pth")
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
    model = build_model()
    model.load_state_dict(torch.load("best_online_soc.pth", map_location=device))
    model.eval()
    
    # Test auf den Validierungsdaten
    print("\n=== Test on Validation Data ===")
    val_loss, preds_val, gts_val = evaluate_online(model, df_val, device, desc="Testing on Val Data")
    timestamps_val = df_val['timestamp'].values
    mae_val = np.mean(np.abs(preds_val - gts_val))
    rmse_val = np.sqrt(np.mean((preds_val - gts_val)**2))
    print(f"Validation Data - MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")
    
    # Plot für Validierungsdaten
    plt.figure(figsize=(10,4))
    plt.plot(timestamps_val, gts_val, 'k-', label="Ground Truth")
    plt.plot(timestamps_val, preds_val, 'r-', label="Prediction")
    plt.title(f"Final Test on Validation Data ({len(df_val)} samples)")
    plt.xlabel("Zeit")
    plt.ylabel("SOC")
    plt.legend()
    plt.annotate(f"MAE: {mae_val:.4f}\nRMSE: {rmse_val:.4f}", xy=(0.01,0.95), xycoords='axes fraction', va='top')
    plt.tight_layout()
    plt.savefig("final_val_test_plot.png")
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

    # Plots für Testdaten
    plt.figure(figsize=(10,4))
    plt.plot(timestamps, gts, 'k-', label="Ground Truth")
    plt.plot(timestamps, preds, 'r-', label="Prediction")
    plt.title(f"Final Test on Test Data ({len(df_test)} samples)")
    plt.xlabel("Zeit")
    plt.ylabel("SOC")
    plt.legend()
    plt.annotate(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}", xy=(0.01,0.95), xycoords='axes fraction', va='top')
    plt.tight_layout()
    plt.savefig("final_test_plot.png")
    plt.close()
    
    zoom_n = min(5000, len(preds))
    for name, seg in [("Start", slice(0, zoom_n)), ("End", slice(-zoom_n, None))]:
        plt.figure(figsize=(10,4))
        plt.plot(timestamps[seg], gts[seg], 'k-', label="Ground Truth")
        plt.plot(timestamps[seg], preds[seg], 'r-', label="Prediction")
        plt.title(f"Zoom {name} - Test Data")
        plt.xlabel("Zeit")
        plt.ylabel("SOC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"zoom_{name.lower()}_test_plot.png")
        plt.close()

if __name__ == "__main__":
    # HIER BATCHSIZE ÄNDERN - nur ein Ort für die Änderung
    BATCH_SIZE = 128  # <<< HIER DIE BATCH-GRÖßE EINSTELLEN
    
    # Für schnelleres Training: größere Batch-Größe und batch-Training statt online-Training
    train_online(epochs=5, lr=1e-3, batch_size=BATCH_SIZE, online_train=False, log_csv_path="training_log.csv")
    test_online(log_csv_path="training_log.csv")
