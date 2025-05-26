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

# Konstanten
SEQ_CHUNK_SIZE = 4096    # Länge der Zeitreihen-Chunks für Seq-to-Seq

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

# RAM-sparende Daten vorbereitung
def load_data_memory_efficient(base_path: str = "/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes"):
    base = Path(base_path)
    
    # Neue Zellkonfiguration
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
            
    # Schritt 3: Validierungsdaten vorbereiten (jeweils 50% für Validierung)
    print("Lade und skaliere Validierungsdaten...")
    val_dfs = {}
    for name in val_cells:
        if name in cells:
            df = cells[name].copy()
            df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
            L = len(df)
            # 50% für Validierung verwenden
            df_val = df.iloc[:int(L * 0.5)].copy()
            df_val[feats] = scaler.transform(df_val[feats])
            val_dfs[name] = df_val
            print(f"[DEBUG] VAL {name}: {len(df_val)} Zeilen")
            
            # Debug: NaN-Check
            nan_counts = df_val[feats].isna().sum().to_dict()
            print(f"[DEBUG] {name} Val NaNs:", {k:v for k,v in nan_counts.items() if v>0} or "none")
    
    # Schritt 4: Testdaten vorbereiten (10% nach den 50% Validierung)
    print("Lade und skaliere Testdaten...")
    test_dfs = {}
    for name in val_cells:
        if name in cells:
            df = cells[name].copy()
            df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
            L = len(df)
            # 10% für Test (50%-60% des Datensatzes)
            i1, i2 = int(L * 0.5), int(L * 0.6)
            df_test = df.iloc[i1:i2].copy()
            df_test[feats] = scaler.transform(df_test[feats])
            test_dfs[name] = df_test
            print(f"[DEBUG] TEST {name}: {len(df_test)} Zeilen")
    
    # RAM freigeben
    del cells
    gc.collect()
    
    return train_scaled, val_dfs, test_dfs, train_cells, val_cells, scaler

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

# Stabilere Weight-initialization 
def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, p in m.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                # Xavier/Glorot für bessere Stabilität
                nn.init.xavier_normal_(p, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(p, 0)
                # LSTM forget gate bias auf 1 setzen für bessere Stabilität
                n = p.size(0)
                start, end = n // 4, n // 2
                p.data[start:end].fill_(1.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# Verbessertes Modell mit höherer Stabilität
def build_model(input_size=4, hidden_size=128, num_layers=2, dropout=0.1, mlp_hidden=256):
    class SOCModel(nn.Module):
        def __init__(self):
            super().__init__()
            # LSTM mit Layer Normalization für bessere Stabilität
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            
            # Stabileres MLP mit Batch Normalization
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.BatchNorm1d(mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden // 2),
                nn.BatchNorm1d(mlp_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden // 2, 1),
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

# Verbesserte Seq-to-Seq-Evaluierung
def evaluate_seq2seq(model, df, device, desc="Evaluation"):
    """Seq-to-Seq-Validation mit Chunking und TQDM."""
    model.eval()
    seq    = df[["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m"]].values
    labels = df["SOC_ZHU"].values
    total = len(seq)
    n_chunks = math.ceil(total / SEQ_CHUNK_SIZE)
    h, c = init_hidden(model, batch_size=1, device=device)
    h, c = h.contiguous(), c.contiguous()
    preds = []

    pbar = tqdm(total=n_chunks, desc=desc, leave=False)
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
            pbar.update(1)
    pbar.close()

    preds = np.array(preds)
    gts = labels[:len(preds)]
    return np.mean((preds - gts) ** 2), preds, gts

# Multi-Validierung: RMSE über mehrere Validierungszellen mitteln
def evaluate_multi_validation(model, val_dfs, device):
    """Evaluiere über alle Validierungszellen und mittele die Ergebnisse"""
    val_rmses = []
    val_results = {}
    
    for name, df_val in val_dfs.items():
        val_mse, val_preds, val_gts = evaluate_seq2seq(model, df_val, device, desc=f"Val {name}")
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
def train_model(epochs=500, lr=3e-4, hidden_size=128, dropout=0.1, 
                log_csv_path="training_log.csv"):
    
    print("=== Lade Daten (RAM-sparsam) ===")
    train_scaled, val_dfs, test_dfs, train_cells, val_cells, scaler = load_data_memory_efficient()
    
    # Debug: Datenzusammenfassung
    print(f"[DEBUG] Chunk size: {SEQ_CHUNK_SIZE}")
    print(f"[DEBUG] Trainingszellen: {train_cells}")
    print(f"[DEBUG] Validierungszellen: {val_cells}")
    
    for name, df in train_scaled.items():
        print(f"[DEBUG] TRAIN {name}: {len(df)} rows")
    for name, df in val_dfs.items():
        print(f"[DEBUG] VAL {name}: {len(df)} rows")
    for name, df in test_dfs.items():
        print(f"[DEBUG] TEST {name}: {len(df)} rows")

    # Plots der Rohdaten
    print("=== Erstelle Datenplots ===")
    for name, df in train_scaled.items():
        plt.figure(figsize=(12,4))
        plt.plot(df['timestamp'], df['SOC_ZHU'], label=name, alpha=0.8)
        plt.title(f"Train SOC {name}")
        plt.xlabel('Zeit')
        plt.ylabel('SOC')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"train_{name}_plot.png", dpi=150)
        plt.close()

    # Val-Plots
    for name, df in val_dfs.items():
        plt.figure(figsize=(12,4))
        plt.plot(df['timestamp'], df['SOC_ZHU'], label=name, alpha=0.8)
        plt.title(f"Validation SOC {name}")
        plt.xlabel('Zeit')
        plt.ylabel('SOC')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"val_{name}_plot.png", dpi=150)
        plt.close()

    print("=== Initialisiere Modell ===")
    model = build_model(hidden_size=hidden_size, dropout=dropout, num_layers=2, mlp_hidden=256)
    
    # Modell-Parameter anzeigen
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modell-Parameter: {total_params:,} total, {trainable_params:,} trainierbar")
    
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, eps=1e-8)
    # Weniger aggressive LR-Reduktion für Stabilität  
    scheduler = ReduceLROnPlateau(optim, mode='min', patience=10, factor=0.7, verbose=True, min_lr=1e-6)
    criterion = nn.MSELoss()
    # Mixed Precision deaktiviert für bessere Stabilität
    use_amp = False
    scaler_amp = GradScaler(enabled=use_amp)
    
    best_val_rmse = float('inf')

    # Logging vorbereiten
    log_fields = ["epoch", "train_rmse", "avg_val_rmse"] + [f"val_rmse_{cell}" for cell in val_cells]
    log_rows = []

    print(f"=== Starte Training für {epochs} Epochen ===")
    
    for ep in range(1, epochs+1):
        model.train()
        total_loss, total_steps = 0.0, 0
        print(f"\n=== Epoch {ep}/{epochs} ===")

        # Training über alle Trainingszellen
        for name, df in train_scaled.items():
            ds = CellDataset(df, SEQ_CHUNK_SIZE)
            print(f"--> {name}, Batches: {len(ds)}")
            dl = DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                num_workers=2,  # Reduziert für RAM-Einsparung
                pin_memory=True
            )
            h, c = init_hidden(model, device=device)
            h, c = h.contiguous(), c.contiguous()
            
            for x_b, y_b in tqdm(dl, desc=f"{name} Ep{ep}", leave=False):
                x_b, y_b = x_b.to(device), y_b.to(device)
                x_b = x_b.contiguous()
                
                optim.zero_grad()
                
                with autocast(device_type=device.type, enabled=use_amp):
                    model.lstm.flatten_parameters()
                    pred, (h, c) = model(x_b, (h, c))
                    
                    # Erweiterte NaN/Inf/Extremwerte-Checks für Stabilität
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        print(f"WARNING: NaN/Inf detected in predictions at epoch {ep}, resetting hidden states")
                        h, c = init_hidden(model, device=device)
                        h, c = h.contiguous(), c.contiguous()
                        continue
                    
                    # Check für extreme Werte
                    if (pred > 1.1).any() or (pred < -0.1).any():
                        print(f"WARNING: Extreme predictions detected at epoch {ep}, clipping values")
                        pred = torch.clamp(pred, 0.0, 1.0)
                    
                    loss = criterion(pred, y_b)
                    
                    # Loss validation mit mehr Checks
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                        print(f"WARNING: Invalid loss ({loss.item()}) at epoch {ep}, skipping batch")
                        continue
                
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optim)
                # Stärkeres gradient clipping für bessere Stabilität
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(optim)
                scaler_amp.update()
                
                h, c = h.detach(), c.detach()
                total_loss += loss.item()   
                total_steps += 1
            
            # RAM nach jeder Zelle aufräumen
            del ds, dl
            gc.collect()

        avg_train_mse = total_loss / total_steps
        train_rmse = math.sqrt(avg_train_mse)
        print(f"Epoch {ep} Training abgeschlossen, train RMSE={train_rmse:.6f}")

        # Multi-Validierung
        print("=== Validierung ===")
        avg_val_rmse, val_results = evaluate_multi_validation(model, val_dfs, device)
        
        # Validierungsplots alle 10 Epochen
        if ep % 10 == 0 or ep <= 5:
            for name, result in val_results.items():
                plt.figure(figsize=(12,4))
                plt.plot(result['df']['timestamp'], result['gts'], 'k-', label="Ground Truth", alpha=0.8)
                plt.plot(result['df']['timestamp'], result['preds'], 'r-', label="Prediction", alpha=0.8)
                plt.title(f"Validierung {name} Ep{ep} — RMSE: {result['rmse']:.4f}")
                plt.xlabel('Zeit')
                plt.ylabel('SOC')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"val_{name}_epoch{ep:03d}.png", dpi=150)
                plt.close()

        scheduler.step(avg_val_rmse)
        
        # Best Model speichern
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Neues bestes Modell gespeichert! Val RMSE: {best_val_rmse:.6f}")

        # Logging
        log_row = {
            "epoch": ep, 
            "train_rmse": train_rmse, 
            "avg_val_rmse": avg_val_rmse
        }
        for cell in val_cells:
            if cell in val_results:
                log_row[f"val_rmse_{cell}"] = val_results[cell]['rmse']
        
        log_rows.append(log_row)

        # CSV nach jeder Epoche aktualisieren
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
            writer.writerows(log_rows)
        
        # RAM aufräumen
        gc.collect()

    print(f"=== Training abgeschlossen! Beste Val RMSE: {best_val_rmse:.6f} ===")
    return train_rmse, avg_val_rmse

# Test-Funktion
def test_model(log_csv_path="training_log.csv"):
    """Teste das beste Modell auf allen Testzellen"""
    print("=== Lade Testdaten ===")
    _, _, test_dfs, _, val_cells, _ = load_data_memory_efficient()
    
    print("=== Lade bestes Modell ===")
    model = build_model(hidden_size=128, dropout=0.01, num_layers=2, mlp_hidden=256)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))    
    model.eval()

    test_results = {}
    test_rmses = []
    
    print("=== Teste auf allen Testzellen ===")
    for name, df_test in test_dfs.items():
        test_mse, test_preds, test_gts = evaluate_seq2seq(model, df_test, device, desc=f"Test {name}")
        test_rmse = math.sqrt(test_mse)
        test_mae = np.mean(np.abs(test_preds - test_gts))
        test_rmses.append(test_rmse)
        
        test_results[name] = {
            'rmse': test_rmse,
            'mae': test_mae,
            'preds': test_preds,
            'gts': test_gts,
            'df': df_test
        }
        
        print(f"Test {name}: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")
        
        # Test-Plot
        plt.figure(figsize=(12,4))
        plt.plot(df_test['timestamp'], test_gts, 'k-', label="Ground Truth", alpha=0.8)
        plt.plot(df_test['timestamp'], test_preds, 'r-', label="Prediction", alpha=0.8)
        plt.title(f"Test {name} — MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
        plt.xlabel('Zeit')
        plt.ylabel('SOC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"test_{name}_final.png", dpi=150)
        plt.close()
    
    # Mittlere Testperformance
    avg_test_rmse = np.mean(test_rmses)
    print(f"\nMittlere Test RMSE über alle Zellen: {avg_test_rmse:.4f}")
    
    # Logging der Testresultate
    try:
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["Test Results"])
            writer.writerow(["cell", "test_mae", "test_rmse"])
            for name, result in test_results.items():
                writer.writerow([name, result['mae'], result['rmse']])
            writer.writerow(["AVERAGE", "", avg_test_rmse])
    except Exception as e:
        print(f"Fehler beim Schreiben der Testresultate: {e}")

    return test_results, avg_test_rmse

def create_summary_plots():
    """Erstelle zusammenfassende Plots der Trainingsergebnisse"""
    try:
        # Lade Trainingslog
        df_log = pd.read_csv("training_log.csv")
        
        # Trainings- und Validierungskurven
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(df_log['epoch'], df_log['train_rmse'], 'b-', label='Train RMSE', alpha=0.8)
        plt.plot(df_log['epoch'], df_log['avg_val_rmse'], 'r-', label='Avg Val RMSE', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Training und Validierung RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        # Einzelne Validierungszellen
        val_cols = [col for col in df_log.columns if col.startswith('val_rmse_')]
        for col in val_cols:
            cell_name = col.replace('val_rmse_', '')
            plt.plot(df_log['epoch'], df_log[col], label=f'Val {cell_name}', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Einzelne Validierungszellen RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_summary.png', dpi=150)
        plt.close()
        
        print("Zusammenfassende Plots erstellt: training_summary.png")
        
    except Exception as e:
        print(f"Fehler beim Erstellen der Zusammenfassungsplots: {e}")

if __name__ == "__main__":
    print("=== BMS SOC LSTM Training v1.2.3.6 ===")
    print("Training-Konfiguration:")
    print("- Epochen: 500")
    print("- Learning Rate: 1e-3") 
    print("- Hidden Size: 128")
    print("- Dropout: 0.01")
    print("- Num Layers: 2")
    print("- MLP Hidden: 256")
    print("- Kein Early Stopping")
    
    # Training mit stabileren Parametern
    train_rmse, val_rmse = train_model(
        epochs=500,
        lr=5e-4,  # Reduzierte Learning Rate für Stabilität
        hidden_size=128,
        dropout=0.05,  # Erhöhtes Dropout gegen Instabilität
        log_csv_path="training_log.csv"
    )
    
    # Test
    test_results, avg_test_rmse = test_model(log_csv_path="training_log.csv")
    
    # Zusammenfassende Plots
    create_summary_plots()
    
    print("\n=== Finale Ergebnisse ===")
    print(f"Finale Train RMSE: {train_rmse:.6f}")
    print(f"Finale Val RMSE: {val_rmse:.6f}")
    print(f"Finale Test RMSE: {avg_test_rmse:.6f}")
    print("Training abgeschlossen!")
