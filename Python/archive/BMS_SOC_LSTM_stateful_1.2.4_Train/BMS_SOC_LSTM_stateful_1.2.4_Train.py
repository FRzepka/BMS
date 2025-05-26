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
from sklearn.preprocessing import MaxAbsScaler, StandardScaler # Updated import
from torch.utils.data import Dataset, DataLoader
import csv
import math
import itertools
import pickle  

# Konstanten
SEQ_CHUNK_SIZE      = 10000    # Länge der Zeitreihen-Chunks (Standard)
USE_FULL_SEQUENCE   = True   # True → pro Zelle nur 1 Batch mit voller Länge
HIDDEN_SIZE         = 32
MLP_HIDDEN          = 32

# Gerät auswählen und cuDNN optimieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Datenlade-Funktion
def load_cell_data(data_dir: Path):
    cell_file_paths = {}
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("MGFarm_18650_"):
            dfp = folder / "df.parquet"
            if dfp.exists():
                cell_file_paths[folder.name] = dfp
            else:
                print(f"Warning: {dfp} fehlt")
    return cell_file_paths

# Daten vorbereiten
def load_data(base_path: str = "/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes"):
    base = Path(base_path)
    # cell_paths ist jetzt ein Dict von cell_name zu Path-Objekt der Parquet-Datei
    cell_paths = load_cell_data(base)
    
    # neue Trainingszellen und feste Validierungszelle
    train_cells = [f"MGFarm_18650_C{str(i).zfill(2)}" for i in [1,3,5,9,11,13,19,21,23,25,27]]
    val_cells = ["MGFarm_18650_C07","MGFarm_18650_C15","MGFarm_18650_C17"]
    # Feature-Liste
    feats = ["Voltage[V]","Current[A]","SOH_ZHU","Q_m"]

    # StandardScaler iterativ über alle Zellen fitten, ohne alle gleichzeitig zu laden
    print("[INFO] StandardScaler wird iterativ über alle Zellen gefittet...")
    scaler = StandardScaler()

    for cell_name, df_path in cell_paths.items():
        # Lade nur die benötigten Spalten temporär, um Speicher zu sparen
        # und für partial_fit
        df_temp = pd.read_parquet(df_path, columns=feats)
        # Entferne Zeilen mit NaN-Werten in den relevanten Spalten, da StandardScaler diese nicht mag
        # und partial_fit sonst Fehler werfen kann oder falsche Statistiken lernt.
        original_len_temp = len(df_temp)
        df_temp.dropna(subset=feats, inplace=True)
        if len(df_temp) < original_len_temp:
            print(f"Info: Beim Fitten des Scalers wurden {original_len_temp - len(df_temp)} Zeilen mit NaNs aus {cell_name} entfernt.")
        
        if not df_temp.empty:
            scaler.partial_fit(df_temp[feats])
        # df_temp wird hier implizit aus dem Speicher entfernt

    print(f"[INFO] StandardScaler fitten abgeschlossen.")
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        print(f"[INFO] Mean: {scaler.mean_}, Scale (StdDev): {scaler.scale_}")
    else:
        print("[INFO] Scaler wurde nicht gefittet, möglicherweise waren alle DataFrames leer oder enthielten nur NaNs.")


    # Skalierte Trainingsdaten
    train_scaled = {}
    for name in train_cells:
        if name not in cell_paths:
            print(f"Warning: Trainingszelle {name} nicht in cell_paths gefunden. Überspringe.")
            continue
        df_path = cell_paths[name]
        df = pd.read_parquet(df_path) # Lade den vollen DataFrame jetzt
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        
        original_len = len(df)
        df.dropna(subset=feats, inplace=True) 
        if len(df) < original_len:
            print(f"Warning: {original_len - len(df)} Zeilen mit NaNs in Features für Trainingszelle {name} entfernt vor dem Transformieren.")

        if not df.empty:
            if hasattr(scaler, 'mean_'): # Prüfen ob Scaler gefittet wurde
                df[feats] = scaler.transform(df[feats])
                train_scaled[name] = df
            else:
                print(f"Warning: Scaler nicht gefittet, Transformation für Trainingszelle {name} übersprungen.")
                # Optional: die unskalierten Daten hinzufügen oder Fehler werfen
                # train_scaled[name] = df 
        else:
            print(f"Warning: DataFrame für Trainingszelle {name} ist nach NaN-Entfernung leer. Überspringe.")

    # debug: check for NaNs after scaling
    for name, df2 in train_scaled.items():
        nan_counts = pd.DataFrame(df2[feats]).isna().sum().to_dict()
        print(f"[DEBUG] {name} NaNs after train scaling:", {k:v for k,v in nan_counts.items() if v>0} or "none")

    # vollständige Validierung auf allen drei Zellen
    df_vals = {}
    for name in val_cells:
        if name not in cell_paths:
            print(f"Warning: Validierungszelle {name} nicht in cell_paths gefunden. Überspringe.")
            continue
        df_path = cell_paths[name]
        dfv = pd.read_parquet(df_path) # Lade den vollen DataFrame jetzt
        dfv['timestamp'] = pd.to_datetime(dfv['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        
        original_len_val = len(dfv)
        dfv.dropna(subset=feats, inplace=True) 
        if len(dfv) < original_len_val:
            print(f"Warning: {original_len_val - len(dfv)} Zeilen mit NaNs in Features für Validierungszelle {name} entfernt vor dem Transformieren.")
            
        if not dfv.empty:
            if hasattr(scaler, 'mean_'): # Prüfen ob Scaler gefittet wurde
                dfv[feats] = scaler.transform(dfv[feats])
                df_vals[name] = dfv
            else:
                print(f"Warning: Scaler nicht gefittet, Transformation für Validierungszelle {name} übersprungen.")
                # df_vals[name] = dfv
        else:
            print(f"Warning: DataFrame für Validierungszelle {name} ist nach NaN-Entfernung leer. Überspringe.")
            
    return train_scaled, df_vals, train_cells, val_cells, scaler

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

# Modell: LSTM + Dropout + MLP-Head (verwendet globale HIDDEN_SIZE und MLP_HIDDEN)
def build_model(input_size=4, num_layers=1, dropout=0.1):
    class SOCModel(nn.Module):
        def __init__(self):
            super().__init__()
            # LSTM ohne Dropout (voller Informationsfluss)
            self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, num_layers,
                                batch_first=True, dropout=0.0)
            # hidden_size bestimmt die Dim. der LSTM-Ausgabe
            # mlp_hidden ist die Größe der verborgenen MLP-Schicht
            # deeper MLP-Head
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
            self.lstm.flatten_parameters()       # Ensure parameters are contiguous for LSTM
            x = x.contiguous()                   # Ensure input data is contiguous
            
            h_in, c_in = hidden
            h_in, c_in = h_in.contiguous(), c_in.contiguous() # Ensure hidden states are contiguous
            hidden_contiguous = (h_in, c_in)
            
            # If USE_FULL_SEQUENCE is True, this implies potentially very long sequences for training.
            # Disable cuDNN for the LSTM layer in this case, similar to how
            # evaluate_onechunk_seq2seq handles long sequences.
            # The evaluate_onechunk_seq2seq function has its own top-level cuDNN disabling context
            # for the entire model call, which will also result in non-cuDNN LSTM execution.
            if USE_FULL_SEQUENCE:
                with torch.backends.cudnn.flags(enabled=False):
                    out, hidden_out = self.lstm(x, hidden_contiguous)
            else:
                out, hidden_out = self.lstm(x, hidden_contiguous)
                
            batch, seq_len, hid = out.size()
            out_flat = out.contiguous().view(batch * seq_len, hid) # Flatten output for MLP
            soc_flat = self.mlp(out_flat)
            soc = soc_flat.view(batch, seq_len) # Reshape to batch, seq_len
            return soc, hidden_out
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
        # model.lstm.flatten_parameters() # REMOVED: Handled in model.forward()
        # disable cuDNN hier, um sehr lange Ein-Chuck-Sequenzen zu erlauben
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = model(chunk, (h, c))
    preds = out.squeeze(0).cpu().numpy()
    mse   = np.mean((preds - labels)**2)
    return mse, preds, labels

# Training Funktion mit Batch-Training und Seq2Seq-Validierung
def train_online(
    epochs=500, lr=1e-4, online_train=False,
    dropout=0.1852, patience=300,
    log_csv_path="training_log.csv", out_dir="training_run",
    train_data=None, df_vals=None, feature_scaler=None):

    # convert out_dir to Path so "/" operator works
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_csv_path = out_dir / log_csv_path

    if train_data is None:
        train_scaled, df_vals, train_cells, val_cells, feature_scaler = load_data()
    else:
        train_scaled = train_data
        # reuse globals
        train_cells = train_cells_glob
        val_cells   = val_cells_glob
        feature_scaler = feature_scaler
    print(f"[INFO] Train cells={train_cells}, Val/Test cells={val_cells}")

    # baue Modell mit globalen Größen
    model = build_model(dropout=dropout)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optim, T_max=50)
    criterion = nn.MSELoss()
    gradient_scaler = GradScaler(enabled=(device.type=="cuda"))
    best_val = float('inf'); no_improve = 0

    # HISTORY & LOG INITIALIZATION
    train_rmse_history = []
    val_rmse_history   = {name: [] for name in val_cells}
    log_rows = []

    for ep in range(1, epochs+1):
        print(f"\n--- Epoch {ep}/{epochs} ---")
        model.train()
        total_loss, steps = 0, 0

        for name, df in train_scaled.items():
            print(f"[Epoch {ep}] Training Cell {name}")
            if not online_train:
                # wähle Sequenz-Länge global
                seq_len = len(df) if USE_FULL_SEQUENCE else SEQ_CHUNK_SIZE
                ds = CellDataset(df, sequence_length=seq_len)
                dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
                h, c = init_hidden(model, batch_size=1, device=device)
                for x_b, y_b in dl:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    x_b = x_b.contiguous()  # Ensure contiguous input
                    
                    optim.zero_grad()
                    
                    # Use proper precision context
                    with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                        # model.lstm.flatten_parameters()  # REMOVED: Handled in model.forward()
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

        # chunk‐basierte Validierung über alle drei Zellen
        val_epoch = {}
        for name, dfv in df_vals.items():
            mse, _, _ = evaluate_onechunk_seq2seq(model, dfv, device)
            rmse = math.sqrt(mse)
            val_rmse_history[name].append(rmse)
            val_epoch[name] = rmse
            print(f"[Epoch {ep}] val RMSE {name}={rmse:.4f}")
        mean_val_rmse = float(np.mean(list(val_epoch.values())))
        print(f"[Epoch {ep}] mean Val RMSE={mean_val_rmse:.4f}")

        # -- kein Test mehr -- (This comment was from an older version, modern logic below)

        # Early Stopping & Model Save
        if mean_val_rmse < best_val:
            best_val = mean_val_rmse
            no_improve = 0
            # Modell speichern
            torch.save(model.state_dict(), out_dir / "best_model.pth") # Saving only model state_dict as per earlier accepted version
            print(f"[Epoch {ep}] Neues Bestmodell gespeichert mit Val RMSE={best_val:.4f}")
        else:
            no_improve += 1
            print(f"[Epoch {ep}] Keine Verbesserung festgestellt (Best Val RMSE bleibt bei {best_val:.4f})")
            if no_improve >= patience:
                print(f"[INFO] Frühzeitiges Stoppen bei Epoche {ep}, da seit {patience} Epochen keine Verbesserung mehr.")
                break

        # Lernraten-Scheduler
        scheduler.step()

        # Logging
        log_rows.append({
            "epoch": ep,
            "train_rmse": train_rmse,
            **{f"val_rmse_{name}": val_epoch[name] for name in val_cells} # Capturing individual val RMSEs
        })
        if ep % 10 == 0 or ep == epochs: # Log periodically or at the end
            df_log = pd.DataFrame(log_rows)
            df_log.to_csv(log_csv_path, index=False)
            print(f"[INFO] Training Log gespeichert: {log_csv_path}")

    # Lade das beste Modell für die finale Bewertung, falls es gespeichert wurde
    best_model_path = out_dir / "best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path)) # Load the best model
        print(f"[INFO] Bestes Modell von {best_model_path} geladen für finale Plots.")
    else:
        print("[INFO] Kein Bestmodell zum Laden gefunden. Plots basieren auf dem letzten Zustand.")


    # Plotte die RMSE-Verläufe
    plt.figure(figsize=(12, 6))
    # Training RMSE
    plt.subplot(1, 2, 1)
    if log_rows: # Ensure log_rows is not empty
        plt.plot([row['epoch'] for row in log_rows], [row['train_rmse'] for row in log_rows], label='Train RMSE', color='blue')
    plt.title('Training RMSE Verlauf')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid()
    plt.legend()

    # Validation RMSE
    plt.subplot(1, 2, 2)
    if log_rows: # Ensure log_rows is not empty
        for name in val_cells:
            # Check if the key exists for all epochs, useful if a val cell was skipped
            if f'val_rmse_{name}' in log_rows[0]:
                plt.plot(
                    [row['epoch'] for row in log_rows],
                    [row[f'val_rmse_{name}'] for row in log_rows],
                    label=f'Val RMSE {name}', linestyle='--'
                )
    plt.title('Validation RMSE Verlauf')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid()
    plt.legend()
    
    plot_path = out_dir / "rmse_verlauf.png"
    plt.savefig(plot_path)
    print(f"[INFO] RMSE-Verlauf gespeichert unter {plot_path}")
    plt.show()


    # Validation SOC-Estimation für alle Zellen (jeweils eigenes Fenster)
    # basierend auf dem besten geladenen Modell
    for name, dfv in df_vals.items():
        if dfv.empty:
            print(f"Skipping SOC estimation plot for {name} as its dataframe is empty.")
            continue
        mse, preds, labels = evaluate_onechunk_seq2seq(model, dfv, device)
        n_samples_to_plot = min(len(labels), int(len(labels) * 0.1) if len(labels) > 100 else len(labels)) # Plot 10% or all if too short
        if n_samples_to_plot == 0:
            print(f"Skipping SOC estimation plot for {name} due to zero samples after processing.")
            continue
            
        plt.figure(figsize=(6,4))
        plt.plot(range(n_samples_to_plot), labels[:n_samples_to_plot], label="true")
        plt.plot(range(n_samples_to_plot), preds[:n_samples_to_plot], '--', label="pred")
        plt.xlabel("Timestep"); plt.ylabel("SOC")
        plt.title(f"Validation SOC estimation (first {n_samples_to_plot} samples) – {name}")
        plt.legend(loc="upper right"); plt.grid()
        soc_plot_path = out_dir/f"val_estimation_{name}.png"
        plt.savefig(soc_plot_path)
        print(f"[INFO] SOC Estimation Plot für {name} gespeichert unter {soc_plot_path}")
        plt.show()
        plt.close()


    return model, feature_scaler, log_rows, df_vals

# Global einmal laden für HPT
train_scaled_glob, df_vals_glob, train_cells_glob, val_cells_glob, feature_scaler_glob = load_data()
print("[INFO] Global data loaded.")

if __name__ == "__main__":
    if train_scaled_glob is not None and df_vals_glob is not None and feature_scaler_glob is not None and train_cells_glob and val_cells_glob:
        # Check if train_scaled_glob or df_vals_glob are empty which can happen if all data was NaN
        if not train_scaled_glob:
            print("[ERROR] Globale Trainingsdaten (train_scaled_glob) sind leer. Training wird nicht gestartet.")
        elif not df_vals_glob:
            print("[ERROR] Globale Validierungsdaten (df_vals_glob) sind leer. Training wird nicht gestartet.")
        else:
            train_online(
                epochs=500, lr=1e-4, # Beispielwerte
                dropout=0.1852, patience=300, # Beispielwerte
                out_dir="training_run",
                train_data=train_scaled_glob,
                df_vals=df_vals_glob,
                feature_scaler=feature_scaler_glob
            )
    else:
        print("[ERROR] Globale Daten konnten nicht korrekt geladen werden oder sind unvollständig. Training wird nicht gestartet.")