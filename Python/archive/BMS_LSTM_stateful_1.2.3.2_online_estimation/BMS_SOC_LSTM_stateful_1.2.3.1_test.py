import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast  # Updated import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # besser in reinen Terminalscreens

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

    feats = ["Voltage[V]", "Current[A]"]
    # Trainingsdaten laden und Timestamp
    train_dfs = {}
    for name in train_cells:
        df = cells[name].copy()
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        train_dfs[name] = df

    # Skalar fitten (Min-Max zwischen 0 und 1)
    df_all_train = pd.concat(train_dfs.values(), ignore_index=True)
    scaler = MinMaxScaler(feature_range=(0,1)).fit(df_all_train[feats])

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
    i1 = int(L*0.4)
    i2 = i1 + int(L*0.1)
    df_val = df3.iloc[:i1].copy()
    df_test = df3.iloc[i1:i2].copy()
    df_val[feats] = scaler.transform(df_val[feats])
    df_test[feats] = scaler.transform(df_test[feats])

    return train_scaled, df_val, df_test, train_cells, val_cell

# Entferne WindowedDataset-Klasse, da sie nicht mehr benötigt wird
# und behalte nur CellDataset

# Angepasstes Dataset für ganze Zellen
class CellDataset(Dataset):
    def __init__(self, df, batch_size=32):
        """Dataset für eine ganze Zelle, aufgeteilt in Batches für effizientes Training
        
        Args:
            df: DataFrame mit den Zelldaten
            batch_size: Größe der Batches innerhalb der Zelle
        """
        self.data = df[["Voltage[V]", "Current[A]"]].values
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

# Helper-Funktion für die Initialisierung der hidden states
def init_hidden(model, batch_size=1, device=None):
    if device is None:
        device = next(model.parameters()).device
    h = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size, device=device)
    c = torch.zeros_like(h)
    return h, c

# Neuer Helper für Validierung (verarbeitet die gesamte Validierungszelle)
# Entferne seq_len Parameter
def evaluate(model, df_val, batch_size, device):
    model.eval()
    ds_val = CellDataset(df_val, batch_size)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, 
                        num_workers=4, pin_memory=True)
    
    criterion = nn.MSELoss()
    # Hidden state init (nur einmal für die gesamte Val-Zelle)
    h, c = init_hidden(model, device=device)
    
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for x_batch, y_batch in dl_val:
            # x_batch shape: [1, batch_size, 2]
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred, (h, c) = model(x_batch, (h, c))
            loss = criterion(pred, y_batch)
            total_loss += loss.item()
            steps += 1
            # h, c bleiben erhalten (stateful)
    
    return total_loss / steps if steps>0 else float('inf')

# Training mit ganzen Zellen, statefull LSTM
# Entferne seq_len Parameter
def train_online(epochs=10, lr=1e-3, batch_size=1024):
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
    scaler = GradScaler('cuda')  # Updated syntax
    best_val_loss = float('inf')

    for ep in range(1, epochs+1):
        model.train()
        total_loss, total_steps = 0.0, 0
        print(f"\n=== Epoch {ep}/{epochs} ===")
        
        # Für jede Zelle
        for name, df in train_scaled.items():
            # Erstelle Dataset für die gesamte Zelle
            ds = CellDataset(df, batch_size)
            print(f"--> {name}, Batches: {len(ds)}")
            dl = DataLoader(
                ds,
                batch_size=1,  # Immer 1, da jedes Item bereits ein Batch ist
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Hidden States für diese Zelle initialisieren (stateful pro Zelle)
            h, c = init_hidden(model, device=device)
            
            # Verarbeite die Zelle in Batches
            for x_b, y_b in tqdm(dl, desc=f"{name} Ep{ep}", leave=True):
                x_b, y_b = x_b.to(device), y_b.to(device)  # x_b: [1, batch_size, 2]
                optim.zero_grad()
                
                with autocast('cuda'):  # Updated syntax
                    pred, (h, c) = model(x_b, (h, c))
                    loss = criterion(pred, y_b)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optim)
                scaler.update()
                
                # h, c bleiben erhalten für den nächsten Batch (stateful)
                # MUSS detach haben, sonst backward-graph error
                h, c = h.detach(), c.detach()
                
                total_loss += loss.item()
                total_steps += 1
                
        avg_loss = total_loss / total_steps
        print(f"Epoch {ep} Training abgeschlossen, avg train loss={avg_loss:.6f}")

        # Validierung nach jeder Epoche (ohne seq_len Parameter)
        val_loss = evaluate(model, df_val, batch_size, device)
        print(f"Epoch {ep} Validation abgeschlossen, val loss={val_loss:.6f}")
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_online_soc.pth")
            print(f"  Neuer Best-Checkpoint (val loss={val_loss:.6f}) gespeichert")

# Test: Schritt-für-Schritt Streaming-Inferenz ohne Fensterpuffer
def test_online():
    _, df_val, df_test, train_cells, val_cell = load_data()
    print("Test auf Zelle:", val_cell)
    model = build_model()
    model.load_state_dict(torch.load("best_online_soc.pth", map_location=device))
    model.eval()

    # Verwende init_hidden Hilfsfunktion
    h, c = init_hidden(model, device=device)
    preds, gts = [], []
    ds = df_test
    # TQDM-Balken im Test
    for v, i in tqdm(zip(ds['Voltage[V]'].values, ds['Current[A]'].values),
                      total=len(ds), desc="Testing"):
        x = torch.tensor([[v, i]], dtype=torch.float32, device=device).view(1,1,2)
        pred, (h, c) = model(x, (h, c))
        preds.append(pred.item())
        gts.append(ds['SOC_ZHU'].iloc[len(preds)-1])
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

def test_only():
    # Testdaten laden
    _, df_val, df_test, train_cells, val_cell = load_data()
    print("Test auf Zelle:", val_cell)
    model = build_model()
    model.load_state_dict(torch.load("best_online_soc.pth", map_location=device))
    model.eval()

    h = torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size, device=device)
    c = torch.zeros_like(h)
    preds, gts = [], []
    ds = df_test
    for v, i in tqdm(zip(ds['Voltage[V]'].values, ds['Current[A]'].values),
                      total=len(ds), desc="Testing"):
        x = torch.tensor([[v, i]], dtype=torch.float32, device=device).view(1,1,2)
        pred, (h, c) = model(x, (h, c))
        preds.append(pred.item())
        gts.append(ds['SOC_ZHU'].iloc[len(preds)-1])
    preds, gts = np.array(preds), np.array(gts)
    timestamps = df_test['timestamp'].values
    mae = np.mean(np.abs(preds - gts)); rmse = np.sqrt(np.mean((preds - gts)**2))
    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # Ordner "test" anlegen falls nicht vorhanden
    out_dir = Path(__file__).parent / "test"
    out_dir.mkdir(exist_ok=True)

    # Plots wie gehabt, aber im test-Ordner speichern
    plt.figure(figsize=(10,4))
    plt.plot(timestamps, gts, 'k-', label="GT"); plt.plot(timestamps, preds, 'r-')
    plt.title("Online Final Test"); plt.annotate(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}", xy=(0.01,0.95), xycoords='axes fraction', va='top')
    plt.tight_layout(); plt.savefig(out_dir / "final_online_plot.png"); plt.close()
    zoom_n = min(50000, len(preds))
    for name, seg in [("Start", slice(0, zoom_n)), ("End", slice(-zoom_n, None))]:
        plt.figure(figsize=(10,4))
        plt.plot(timestamps[seg], gts[seg], 'k-'); plt.plot(timestamps[seg], preds, 'r-')
        plt.title(f"Zoom {name}"); plt.tight_layout(); plt.savefig(out_dir / f"zoom_{name.lower()}_online_plot.png"); plt.close()

if __name__ == "__main__":
    test_only()
