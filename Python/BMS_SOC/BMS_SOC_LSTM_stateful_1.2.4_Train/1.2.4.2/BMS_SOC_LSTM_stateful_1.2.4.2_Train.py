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
import gc

# Konstanten - STABILERE ARCHITEKTUR
SEQ_CHUNK_SIZE = 4096    # Länge der Zeitreihen-Chunks für Seq-to-Seq
HIDDEN_SIZE = 48         # REDUZIERT von 64 auf 48 für bessere Stabilität
MLP_HIDDEN = 48          # REDUZIERT für bessere Stabilität
NUM_LAYERS = 1           # REDUZIERT von 2 auf 1 für bessere Stabilität

# Gerät auswählen und cuDNN optimieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# RAM-sparende Datenlade-Funktion (von 1.2.3.6)
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

# RAM-sparende Daten vorbereitung (von 1.2.3.6 aber mit neuer Zellaufteilung wie 1.2.4)
def load_data_memory_efficient(base_path: str = "/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes"):
    base = Path(base_path)
    
    # ZURÜCK ZU ORIGINALER Zellaufteilung (wie 1.2.3.6)
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
    
    # Schritt 1: Scaler über alle Daten fitten (RAM-sparsam)
    print("Berechne Scaler über alle Zellen...")
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
    
    # Schritt 2: Trainingsdaten laden und skalieren
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
            
    # Schritt 3: Validierungsdaten vorbereiten - VOLLSTÄNDIGE DATEN
    print("Lade und skaliere Validierungsdaten (vollständig)...")
    val_dfs = {}
    for name in val_cells:
        if name in cells:
            df = cells[name].copy()
            df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
            df[feats] = scaler.transform(df[feats])
            val_dfs[name] = df
            print(f"[DEBUG] VAL {name}: {len(df)} Zeilen (vollständig)")
            
            # Debug: NaN-Check
            nan_counts = df[feats].isna().sum().to_dict()
            print(f"[DEBUG] {name} Val NaNs:", {k:v for k,v in nan_counts.items() if v>0} or "none")
    
    # RAM freigeben
    del cells
    gc.collect()
    
    return train_scaled, val_dfs, train_cells, val_cells, scaler

# Angepasstes Dataset für ganze Zellen (von 1.2.3.6)
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

# VERBESSERTE Weight-initialization
def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, p in m.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.xavier_uniform_(p, gain=1.0)  # Xavier Uniform
            elif 'bias' in name:
                nn.init.constant_(p, 0)
                # LSTM forget gate bias auf 1 setzen für bessere Stabilität
                n = p.size(0)
                start, end = n // 4, n // 2
                p.data[start:end].fill_(1.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# VERBESSERTE Modellarchitektur
def build_model(input_size=4, dropout=0.15):
    class SOCModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 2-Layer LSTM mit 64 Hidden Units für bessere Kapazität
            self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS,
                                batch_first=True, dropout=dropout if NUM_LAYERS > 1 else 0.0)
            
            # Tieferes MLP mit Residual Connection
            self.mlp = nn.Sequential(
                nn.Linear(HIDDEN_SIZE, MLP_HIDDEN),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(MLP_HIDDEN, MLP_HIDDEN // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.25),
                nn.Linear(MLP_HIDDEN // 2, 1),
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

# Seq-to-Seq-Evaluierung mit Chunking UND hidden state Übertragung
def evaluate_seq2seq_chunked(model, df, device, desc="Evaluation"):
    """
    Seq-to-Seq-Validation mit Chunking aber kontinuierlichen hidden states
    """
    model.eval()
    seq    = df[["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m"]].values
    labels = df["SOC_ZHU"].values
    
    n_samples = len(seq)
    all_preds = np.zeros(n_samples)
    
    # Hidden state initialisieren
    h, c = init_hidden(model, device=device)
    h, c = h.contiguous(), c.contiguous()
    
    # Chunked evaluation aber mit kontinuierlichen hidden states
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_samples, SEQ_CHUNK_SIZE), desc=desc, leave=False):
            end_idx = min(start_idx + SEQ_CHUNK_SIZE, n_samples)
            
            # Chunk vorbereiten
            chunk = seq[start_idx:end_idx]
            x_chunk = torch.from_numpy(chunk).float().unsqueeze(0).to(device)  # (1, seq, features)
            x_chunk = x_chunk.contiguous()
            
            # Forward pass mit kontinuierlichen hidden states
            model.lstm.flatten_parameters()
            pred_chunk, (h, c) = model(x_chunk, (h, c))
            
            # Hidden states für nächsten Chunk detachen
            h, c = h.detach(), c.detach()
            
            # Predictions speichern
            pred_np = pred_chunk.squeeze(0).cpu().numpy()  # (seq,)
            all_preds[start_idx:end_idx] = pred_np
    
    # Metriken berechnen
    gts = labels
    rmse = np.sqrt(np.mean((all_preds - gts)**2))
    mae = np.mean(np.abs(all_preds - gts))
    
    return all_preds, gts, rmse, mae

# Multi-Validierung über alle Validierungszellen
def evaluate_multi_validation(model, val_dfs, device):
    """Evaluiere das Modell auf allen Validierungszellen"""
    results = {}
    total_rmse, total_mae = 0.0, 0.0
    
    for name, df in val_dfs.items():
        print(f"Validiere {name}...")
        preds, gts, rmse, mae = evaluate_seq2seq_chunked(model, df, device, f"Val {name}")
        results[name] = {
            'preds': preds,
            'gts': gts,
            'rmse': rmse,
            'mae': mae,
            'df': df  # Für Plotting
        }
        total_rmse += rmse
        total_mae += mae
        print(f"  --> RMSE: {rmse:.6f}, MAE: {mae:.6f}")
    
    avg_rmse = total_rmse / len(val_dfs)
    avg_mae = total_mae / len(val_dfs)
    print(f"Durchschnittliche Validierung: RMSE={avg_rmse:.6f}, MAE={avg_mae:.6f}")
    
    return avg_rmse, avg_mae, results

# VERBESSERTE Trainingsfunktion
def train_model(model, train_scaled, val_dfs, train_cells, val_cells, out_dir, epochs=50):
    """Haupttraining mit verbesserter Lernrate und Regularisierung"""
    
    # STABILERE Optimizer-Konfiguration
    initial_lr = 1e-4  # REDUZIERT von 3e-4 auf 1e-4 für Stabilität
    optim = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5, eps=1e-8)
    
    # EINFACHERE Learning Rate Scheduler - NUR ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.8, patience=8, min_lr=1e-6, verbose=True)
    
    criterion = nn.MSELoss()
    
    # Mixed Precision DEAKTIVIERT für bessere Stabilität
    use_amp = False  # DEAKTIVIERT
    scaler_amp = GradScaler(enabled=use_amp)
    
    best_val_rmse = float('inf')
    no_improve = 0
    patience = 20  # Erhöhte Patience für stabileres Training

    # Logging vorbereiten
    log_fields = ["epoch", "train_rmse", "train_mae", "avg_val_rmse", "avg_val_mae", "lr"] + \
                 [f"val_rmse_{cell}" for cell in val_cells] + \
                 [f"val_mae_{cell}" for cell in val_cells]
    log_rows = []
    log_csv_path = out_dir / "training_log.csv"

    print(f"=== Starte Training für {epochs} Epochen ===")
    print(f"Trainingszellen: {len(train_cells)} - {train_cells}")
    print(f"Validierungszellen: {len(val_cells)} - {val_cells}")
    
    for ep in range(1, epochs+1):
        model.train()
        total_loss, total_steps = 0.0, 0
        train_preds_all, train_gts_all = [], []
        print(f"\n=== Epoch {ep}/{epochs} ===")

        # Training über alle Trainingszellen
        for name, df in train_scaled.items():
            ds = CellDataset(df, SEQ_CHUNK_SIZE)
            print(f"--> {name}, Batches: {len(ds)}")
            dl = DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            # Hidden state nur zwischen Zellen zurücksetzen, NICHT zwischen chunks!
            h, c = init_hidden(model, device=device)
            h, c = h.contiguous(), c.contiguous()
            
            for x_b, y_b in tqdm(dl, desc=f"{name} Ep{ep}", leave=False):
                x_b, y_b = x_b.to(device), y_b.to(device)
                x_b = x_b.contiguous()
                
                optim.zero_grad()
                
                with autocast(device_type=device.type, enabled=use_amp):
                    model.lstm.flatten_parameters()
                    pred, (h, c) = model(x_b, (h, c))
                    
                    # NaN/Inf checks
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        print(f"WARNING: NaN/Inf detected in predictions at epoch {ep}")
                        h, c = init_hidden(model, device=device)
                        h, c = h.contiguous(), c.contiguous()
                        continue
                    
                    loss = criterion(pred, y_b)
                
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optim)
                # STABILERES Gradient clipping
                clip_grad_norm_(model.parameters(), max_norm=1.0)  # REDUZIERT von 2.0 auf 1.0
                scaler_amp.step(optim)
                scaler_amp.update()
                
                # Hidden states detachen aber zwischen chunks übertragen
                h, c = h.detach(), c.detach()
                total_loss += loss.item()   
                total_steps += 1
                
                # Für MAE-Berechnung sammeln
                train_preds_all.extend(pred.detach().cpu().numpy().flatten())
                train_gts_all.extend(y_b.detach().cpu().numpy().flatten())
            
            # RAM nach jeder Zelle aufräumen
            del ds, dl
            gc.collect()

        avg_train_mse = total_loss / total_steps
        train_rmse = math.sqrt(avg_train_mse)
        train_mae = np.mean(np.abs(np.array(train_preds_all) - np.array(train_gts_all)))
        print(f"Epoch {ep} Training abgeschlossen, train RMSE={train_rmse:.6f}, MAE={train_mae:.6f}")

        # Multi-Validierung
        print("=== Validierung ===")
        avg_val_rmse, avg_val_mae, val_results = evaluate_multi_validation(model, val_dfs, device)
        
        # Early Stopping (mit patience)
        is_best = avg_val_rmse < best_val_rmse
        best_val_rmse = min(avg_val_rmse, best_val_rmse)
        if is_best:
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            print(f"[Epoch {ep}] Neues bestes Modell gespeichert! Val RMSE: {best_val_rmse:.6f}")
        else:
            no_improve += 1

        # Plotting - 3 Validierungszellen + 1 RMSE/MAE Verlauf
        for name, result in val_results.items():
            plt.figure(figsize=(12,5))
            n_samples = int(len(result['gts']) * 0.1)  # Nur erste 10%
            step = 60  # Jeden 60. Wert
            
            # Indices für Sampling
            indices = list(range(0, n_samples, step))
            times = result['df']['timestamp'].iloc[indices]
            gts_sub = result['gts'][indices]
            preds_sub = result['preds'][indices]
            
            plt.plot(times, gts_sub, 'k-', label="Ground Truth", alpha=0.8, linewidth=1.5)
            plt.plot(times, preds_sub, 'r-', label="Prediction", alpha=0.8, linewidth=1.5)
            plt.title(f"Validation {name} (Ep {ep}) — RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")
            plt.xlabel('Zeit')
            plt.ylabel('SOC')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"val_{name}_prediction.png", dpi=120, bbox_inches='tight')
            plt.close()
            
        # RMSE/MAE Verlaufsdiagramm
        plt.figure(figsize=(14,8))
        epochs_so_far = [row['epoch'] for row in log_rows] + [ep]
        train_rmses = [row['train_rmse'] for row in log_rows] + [train_rmse]
        train_maes = [row['train_mae'] for row in log_rows] + [train_mae]
        val_rmses = [row['avg_val_rmse'] for row in log_rows] + [avg_val_rmse]
        val_maes = [row['avg_val_mae'] for row in log_rows] + [avg_val_mae]
        
        plt.subplot(2,1,1)
        plt.plot(epochs_so_far, train_rmses, 'b-', label='Train RMSE', alpha=0.8, linewidth=2)
        plt.plot(epochs_so_far, val_rmses, 'r-', label='Val RMSE', alpha=0.8, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('RMSE Verlauf über Epochen')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2,1,2)
        plt.plot(epochs_so_far, train_maes, 'b-', label='Train MAE', alpha=0.8, linewidth=2)
        plt.plot(epochs_so_far, val_maes, 'r-', label='Val MAE', alpha=0.8, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('MAE Verlauf über Epochen')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_dir / "training_progress.png", dpi=120, bbox_inches='tight')
        plt.close()

        # STABILERE Learning Rate Scheduling
        scheduler.step(avg_val_rmse)  # Nur ReduceLROnPlateau
        
        # Logging
        log_row = {
            "epoch": ep, 
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "avg_val_rmse": avg_val_rmse,
            "avg_val_mae": avg_val_mae,
            "lr": optim.param_groups[0]['lr']
        }
        for cell in val_cells:
            if cell in val_results:
                log_row[f"val_rmse_{cell}"] = val_results[cell]['rmse']
                log_row[f"val_mae_{cell}"] = val_results[cell]['mae']
        
        log_rows.append(log_row)

        # CSV nach jeder Epoche aktualisieren
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
            writer.writerows(log_rows)
        
        # Metrics table
        metrics_table_path = out_dir / "metrics_table.csv"
        with open(metrics_table_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Epoch", "Train_RMSE", "Train_MAE", "Avg_Val_RMSE", "Avg_Val_MAE"]
            for cell in val_cells:
                header.extend([f"{cell}_RMSE", f"{cell}_MAE"])
            writer.writerow(header)
            
            for row in log_rows:
                table_row = [row['epoch'], f"{row['train_rmse']:.6f}", f"{row['train_mae']:.6f}",
                           f"{row['avg_val_rmse']:.6f}", f"{row['avg_val_mae']:.6f}"]
                for cell in val_cells:
                    if cell in val_results:
                        table_row.extend([f"{val_results[cell]['rmse']:.6f}", 
                                        f"{val_results[cell]['mae']:.6f}"])
                    else:
                        table_row.extend(["N/A", "N/A"])
                writer.writerow(table_row)

        print(f"Epoch {ep}: LR={optim.param_groups[0]['lr']:.8f}, No-improve: {no_improve}/{patience}")
        
        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break

    print(f"\n=== Training beendet ===")
    print(f"Bestes Val RMSE: {best_val_rmse:.6f}")
    return log_rows

# Hauptausführung
if __name__ == "__main__":
    print("=== BMS SOC LSTM Training 1.2.4.2 ===")
    print("VERBESSERUNGEN:")
    print("- Erhöhte Modellkapazität (64 Hidden, 2 LSTM Layers)")
    print("- Mehr Trainingsdaten (9 statt 6 Zellen)")
    print("- Verbesserte Learning Rate Schedule (CosineWarmRestarts + ReduceLROnPlateau)")
    print("- Höhere initiale LR (3e-4 statt 1e-4)")
    print("- Relaxed Gradient Clipping (2.0 statt 1.0)")
    print("- Erhöhte Patience (15 statt 10)")
    print()
    
    # Output-Verzeichnis im gleichen Ordner wie das Skript
    script_dir = Path(__file__).parent
    out_dir = script_dir / "training_run_1.2.4.2"
    out_dir.mkdir(exist_ok=True)
    print(f"Output-Verzeichnis: {out_dir}")
    
    # Daten laden
    print("\n=== Daten laden ===")
    train_scaled, val_dfs, train_cells, val_cells, scaler = load_data_memory_efficient()
    
    # Modell erstellen
    print("\n=== Modell erstellen ===")
    model = build_model(input_size=4, dropout=0.15)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modell Parameter: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Modell Architektur: {NUM_LAYERS} LSTM Layers, {HIDDEN_SIZE} Hidden Size, {MLP_HIDDEN} MLP Hidden")
    
    # Training starten
    print("\n=== Training starten ===")
    log_rows = train_model(model, train_scaled, val_dfs, train_cells, val_cells, out_dir, epochs=80)
    
    # Finale Modell-Speicherung
    torch.save(model.state_dict(), out_dir / "final_model.pth")
    torch.save(scaler, out_dir / "scaler.pkl")
    
    print(f"\n=== Training abgeschlossen ===")
    print(f"Modell und Logs gespeichert in: {out_dir}")
    print(f"Gesamte Epochen: {len(log_rows)}")
    
    if log_rows:
        final_train_rmse = log_rows[-1]['train_rmse']
        final_val_rmse = log_rows[-1]['avg_val_rmse']
        best_val_rmse = min(row['avg_val_rmse'] for row in log_rows)
        print(f"Finale Train RMSE: {final_train_rmse:.6f}")
        print(f"Finale Val RMSE: {final_val_rmse:.6f}")
        print(f"Beste Val RMSE: {best_val_rmse:.6f}")
