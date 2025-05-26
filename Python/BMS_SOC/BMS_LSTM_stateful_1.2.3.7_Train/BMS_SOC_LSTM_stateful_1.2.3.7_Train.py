import os
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
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
import gc

# Konstanten
SEQ_CHUNK_SIZE = 4096    # Länge der Zeitreihen-Chunks für Seq-to-Seq (beibehalten aus 1.2.3.6)
HIDDEN_SIZE = 32         # Kleineres Modell aus 1.2.4
MLP_HIDDEN = 32          # MLP-Größe aus 1.2.4

# Gerät auswählen und cuDNN optimieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# RAM-sparende Datenlade-Funktion
def load_cell_data_memory_efficient(data_dir: Path, cell_names: list):
    """Lade nur die benötigten Zellen"""
    dataframes = {}
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name in cell_names:
            dfp = folder / "df.parquet"
            if dfp.exists():
                dataframes[folder.name] = pd.read_parquet(dfp)
                print(f"Geladen: {folder.name} - {len(dataframes[folder.name])} Zeilen")
            else:
                print(f"Warning: {dfp} fehlt")
    return dataframes

# Daten vorbereiten (angepasst für vollständige Validierung ohne separaten Test)
def load_data_memory_efficient(base_path: str = "/home/users/f/flo01010010/HPC_projects/5_Data/MGFarm_18650_Dataframes"):
    base = Path(base_path)
    
    # Zellkonfiguration wie in 1.2.3.6
    train_cells = [
        "MGFarm_18650_C01",
        "MGFarm_18650_C03", 
        "MGFarm_18650_C05",
        "MGFarm_18650_C11",
        "MGFarm_18650_C17",
        "MGFarm_18650_C23"
    ]
    val_cells = [
        "MGFarm_18650_C07",
        "MGFarm_18650_C19", 
        "MGFarm_18650_C21"
    ]
    
    # Alle benötigten Zellen laden
    all_cells = train_cells + val_cells
    cells = load_cell_data_memory_efficient(base, all_cells)
    
    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m"]
    
    print("=== RAM-sparende Skalierung über alle Daten ===")
    
    # MaxAbsScaler wie in 1.2.3.6 über alle Zellen fitten
    print("Berechne MaxAbsScaler über alle Zellen...")
    scaler = MaxAbsScaler()
    
    # Iterativ über alle Zellen fitten ohne alles im RAM zu behalten
    for name in all_cells:
        if name in cells:
            df = cells[name].copy()
            df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
            scaler.partial_fit(df[feats])
            print(f"Partial fit für {name}: {len(df)} Zeilen")
            del df  # RAM freigeben
            gc.collect()
    
    print("[DEBUG] scaler.scale_:", dict(zip(feats, scaler.scale_)))
    
    # Trainingsdaten laden und skalieren
    print("Lade und skaliere Trainingsdaten...")
    train_scaled = {}
    for name in train_cells:
        if name in cells:
            df = cells[name].copy()
            df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
            df[feats] = scaler.transform(df[feats])
            train_scaled[name] = df
            
            # Debug: NaN-Check
            nan_counts = df[feats].isna().sum().to_dict()
            print(f"[DEBUG] {name} NaNs after scaling:", {k:v for k,v in nan_counts.items() if v>0} or "none")
            
    # Validierungsdaten vorbereiten (VOLLSTÄNDIGE Zellen wie in 1.2.4)
    print("Lade und skaliere Validierungsdaten (vollständig)...")
    val_dfs = {}
    for name in val_cells:
        if name in cells:
            df = cells[name].copy()
            df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
            df[feats] = scaler.transform(df[feats])
            val_dfs[name] = df  # VOLLSTÄNDIGE Zelle
            print(f"[DEBUG] VAL {name}: {len(df)} Zeilen (vollständig)")
            
            # Debug: NaN-Check
            nan_counts = df[feats].isna().sum().to_dict()
            print(f"[DEBUG] {name} Val NaNs:", {k:v for k,v in nan_counts.items() if v>0} or "none")
    
    # RAM freigeben
    del cells
    gc.collect()
    
    return train_scaled, val_dfs, train_cells, val_cells, scaler

# Angepasstes Dataset für ganze Zellen (beibehalten aus 1.2.3.6)
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

# Weight-initialization wie in 1.2.4 (Xavier Uniform)
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

# Modell: LSTM + MLP-Head (Architektur aus 1.2.4)
def build_model(input_size=4, num_layers=1, dropout=0.1852):
    class SOCModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Kompakte LSTM wie in 1.2.4
            self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, num_layers,
                                batch_first=True, dropout=0.0)
            
            # MLP wie in 1.2.4
            self.mlp = nn.Sequential(
                nn.Linear(HIDDEN_SIZE, MLP_HIDDEN),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(MLP_HIDDEN, 1),
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

# Chunked Seq-to-Seq-Evaluierung (RAM-sparend mit Hidden State Continuity)
def evaluate_seq2seq_chunked(model, df, device, desc="Evaluation"):
    """Seq-to-Seq-Validation mit Chunking und kontinuierlichen Hidden States"""
    model.eval()
    seq    = df[["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m"]].values
    labels = df["SOC_ZHU"].values
    total = len(seq)
    n_chunks = math.ceil(total / SEQ_CHUNK_SIZE)
    h, c = init_hidden(model, batch_size=1, device=device)
    h, c = h.contiguous(), c.contiguous()
    preds = []

    print(f">> {desc} mit {n_chunks} Chunks startet")
    with torch.no_grad():
        for i in range(n_chunks):
            s = i * SEQ_CHUNK_SIZE
            e = min(s + SEQ_CHUNK_SIZE, total)
            chunk = torch.tensor(seq[s:e], dtype=torch.float32, device=device).unsqueeze(0)
            chunk = chunk.contiguous()
            model.lstm.flatten_parameters()
            # disable cuDNN für sehr große Chunks
            with torch.backends.cudnn.flags(enabled=False):
                out, (h, c) = model(chunk, (h, c))
            h, c = h.contiguous(), c.contiguous()  # Hidden States für nächsten Chunk
            preds.extend(out.squeeze(0).cpu().numpy())

    preds = np.array(preds)
    gts = labels[:len(preds)]
    return np.mean((preds - gts) ** 2), preds, gts

# Multi-Validierung: RMSE über mehrere Validierungszellen mitteln
def evaluate_multi_validation(model, val_dfs, device):
    """Evaluiere über alle Validierungszellen und mittele die Ergebnisse"""
    val_rmses = []
    val_results = {}
    
    for name, df_val in val_dfs.items():
        val_mse, val_preds, val_gts = evaluate_seq2seq_chunked(model, df_val, device, desc=f"Val {name}")
        val_rmse = math.sqrt(val_mse)
        val_rmses.append(val_rmse)
        val_results[name] = {
            'rmse': val_rmse,
            'preds': val_preds,
            'gts': val_gts,
            'df': df_val
        }
        print(f"  {name}: RMSE={val_rmse:.6f}")
    
    avg_val_rmse = np.mean(val_rmses)
    print(f"  Mittlere Val RMSE: {avg_val_rmse:.6f}")
    
    return avg_val_rmse, val_results

# Haupttraining-Funktion
def train_model(epochs=500, lr=1e-4, dropout=0.1852, patience=300,
                log_csv_path="training_log.csv", out_dir="training_run"):
    
    # convert out_dir to Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = out_dir / log_csv_path
    
    print("=== Lade Daten (RAM-sparsam) ===")
    train_scaled, val_dfs, train_cells, val_cells, scaler = load_data_memory_efficient()
    
    # Debug: Datenzusammenfassung
    print(f"[DEBUG] Chunk size: {SEQ_CHUNK_SIZE}")
    print(f"[DEBUG] Trainingszellen: {train_cells}")
    print(f"[DEBUG] Validierungszellen: {val_cells}")
    
    for name, df in train_scaled.items():
        print(f"[DEBUG] TRAIN {name}: {len(df)} rows")
    for name, df in val_dfs.items():
        print(f"[DEBUG] VAL {name}: {len(df)} rows")
    
    print("=== Modell Setup ===")
    model = build_model(dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)  # Wie in 1.2.4
    criterion = nn.MSELoss()
    gradient_scaler = GradScaler(enabled=(device.type=="cuda"))  # Mixed Precision aktiviert
    
    # Best model tracking
    best_val_rmse = float('inf')
    no_improve = 0
    
    # History tracking
    train_rmse_history = []
    val_rmse_history = {name: [] for name in val_cells}
    log_rows = []
    
    print("=== Training startet ===")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        model.train()
        
        total_loss = 0.0
        total_steps = 0
        
        # Training über alle Zellen (Chunked mit Hidden State Reset zwischen Zellen)
        for cell_name, df in train_scaled.items():
            print(f"[Epoch {epoch}] Training auf Zelle {cell_name}")
            
            dataset = CellDataset(df, SEQ_CHUNK_SIZE)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
            
            # Hidden State Reset zwischen Zellen
            h, c = init_hidden(model, batch_size=1, device=device)
            h, c = h.contiguous(), c.contiguous()
            
            cell_loss = 0.0
            cell_steps = 0
            
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                
                # Mixed Precision Training
                with autocast(device_type='cuda', enabled=(device.type=="cuda")):
                    pred, (h, c) = model(x, (h, c))
                    # Hidden States für nächsten Chunk beibehalten (kontinuierlich)
                    h, c = h.detach().contiguous(), c.detach().contiguous()
                    loss = criterion(pred, y.unsqueeze(-1))
                
                # Gradient Scaling
                gradient_scaler.scale(loss).backward()
                gradient_scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
                
                cell_loss += loss.item()
                cell_steps += 1
                total_loss += loss.item()
                total_steps += 1
            
            print(f"  {cell_name}: Chunks={len(dataloader)}, Loss={cell_loss/cell_steps:.6f}")
        
        train_rmse = math.sqrt(total_loss / total_steps)
        train_rmse_history.append(train_rmse)
        print(f"[Epoch {epoch}] Train RMSE: {train_rmse:.6f}")
        
        # Validierung
        print(f"[Epoch {epoch}] Validierung...")
        avg_val_rmse, val_results = evaluate_multi_validation(model, val_dfs, device)
        
        # Tracking für jede Validierungszelle
        for name in val_cells:
            val_rmse_history[name].append(val_results[name]['rmse'])
        
        # Early Stopping
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            print(f"[Epoch {epoch}] Neues bestes Modell gespeichert!")
        else:
            no_improve += 1
        
        # Learning Rate Scheduler
        scheduler.step()
        
        # Logging
        log_rows.append({
            "epoch": epoch,
            "train_rmse": train_rmse,
            "avg_val_rmse": avg_val_rmse,
            **{f"val_rmse_{name}": val_results[name]['rmse'] for name in val_cells},
            "lr": optimizer.param_groups[0]['lr'],
            "dropout": dropout
        })
        
        # CSV speichern
        df_log = pd.DataFrame(log_rows)
        df_log.to_csv(log_csv_path, index=False)
        
        # Plots erstellen (wie in 1.2.4, aber reduziert für Performance)
        if epoch % 10 == 0:  # Alle 10 Epochen plotten
            print(f"[Epoch {epoch}] Erstelle Plots...")
            
            # 1. RMSE-Verlauf (Train + alle Val-Zellen)
            plt.figure(figsize=(10,6))
            plt.plot(range(1, len(train_rmse_history)+1), train_rmse_history, 'b-', label="Train RMSE")
            for name in val_cells:
                plt.plot(range(1, len(val_rmse_history[name])+1), val_rmse_history[name], 
                        label=f"Val RMSE {name}")
            plt.xlabel("Epoch")
            plt.ylabel("RMSE")
            plt.title("Training & Validation RMSE")
            plt.legend(loc="upper right")
            plt.grid(True)
            plt.savefig(out_dir / "rmse_plot.png", dpi=100, bbox_inches='tight')
            plt.close()
            
            # 2-4. Validierungs-Predictions für jede Zelle (nur erste 10%, jeden 60. Wert)
            for name, result in val_results.items():
                preds, gts = result['preds'], result['gts']
                # Nur erste 10% plotten, jeden 60. Wert
                n_plot = int(len(preds) * 0.1)
                step = 60
                indices = np.arange(0, n_plot, step)
                
                plt.figure(figsize=(10,6))
                plt.plot(indices, gts[indices], 'g-', label="Ground Truth", alpha=0.7)
                plt.plot(indices, preds[indices], 'r-', label="Prediction", alpha=0.7)
                plt.xlabel("Time Step")
                plt.ylabel("SOC")
                plt.title(f"SOC Prediction {name} (Epoch {epoch})")
                plt.legend(loc="upper right")
                plt.grid(True)
                plt.savefig(out_dir / f"prediction_{name}.png", dpi=100, bbox_inches='tight')
                plt.close()
        
        # Early Stopping Check
        if no_improve >= patience:
            print(f"[Epoch {epoch}] Early stopping nach {patience} Epochen ohne Verbesserung")
            break
    
    # Lade bestes Modell für finale Bewertung
    model.load_state_dict(torch.load(out_dir / "best_model.pth", weights_only=True))
    print("=== Finale Bewertung ===")
    final_avg_val_rmse, final_val_results = evaluate_multi_validation(model, val_dfs, device)
    print(f"Finale durchschnittliche Val RMSE: {final_avg_val_rmse:.6f}")
    
    return model, scaler, log_rows, val_dfs

if __name__ == "__main__":
    model, scaler, logs, val_data = train_model(
        epochs=500,
        lr=1e-4,
        dropout=0.1852,
        patience=300,
        out_dir="training_run"
    )
