import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm                                    # ← hinzugefügt

# --- Konstante Einstellung ---
SEQ_CHUNK_SIZE = 10000
HIDDEN_SIZE    = 32
MLP_HIDDEN     = 32
MODEL_PATH     = Path("/home/florianr/MG_Farm/6_Scripts/BMS/Python/BMS_SOC/BMS_SOC_LSTM_stateful_1.2.4_Train/BMS_SOC_LSTM_stateful_1.2.4.7_Train_CPU/training_run_1/best_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cell_data(data_dir: Path):
    # statt DataFrames returnen wir hier nur die Parquet-Pfade
    cell_paths = {}
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("MGFarm_18650_"):
            p = folder / "df.parquet"
            if p.exists():
                cell_paths[folder.name] = p
            else:
                print(f"Warning: {p} fehlt")
    return cell_paths

def load_data(base_path: str = "/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes"):
    base      = Path(base_path)
    cell_paths= load_cell_data(base)
    feats     = ["Voltage[V]","Current[A]","SOH_ZHU","Q_m"]
    # iteratives Fitten
    scaler = StandardScaler()
    for name, p in cell_paths.items():
        df_tmp = pd.read_parquet(p, columns=feats)
        df_tmp.dropna(subset=feats, inplace=True)
        if not df_tmp.empty:
            scaler.partial_fit(df_tmp[feats])
    print("[INFO] Scaler iterativ gefittet")

    # Validierungszellen
    val_cells = ["MGFarm_18650_C01","MGFarm_18650_C03","MGFarm_18650_C05"]
    df_vals   = {}
    for name in val_cells:
        if name not in cell_paths:
            print(f"Warning: {name} nicht gefunden, übersprungen")
            continue
        df = pd.read_parquet(cell_paths[name])
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        df[feats]      = scaler.transform(df[feats])
        df_vals[name]  = df

    # für Test brauchen wir Trainingsdaten nicht
    return {}, df_vals, [], val_cells, scaler

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

def evaluate_onechunk_seq2seq(model, df, device):
    """
    Seq2Seq-Eval über genau einen Chunk: ganzes df als (1, N, F)-Sequenz.
    """
    model.eval()
    seq    = df[["Voltage[V]","Current[A]","SOH_ZHU","Q_m"]].values
    labels = df["SOC_ZHU"].values
    total = len(seq)
    n_chunks = math.ceil(total / SEQ_CHUNK_SIZE)
    h, c   = init_hidden(model, batch_size=1, device=device)
    h, c   = h.contiguous(), c.contiguous()
    preds = []

    with torch.no_grad():
        for i in tqdm(range(n_chunks), desc="Test chunks"):  # ← tqdm für Fortschritt
            s = i * SEQ_CHUNK_SIZE
            e = min(s + SEQ_CHUNK_SIZE, total)
            chunk = torch.tensor(seq[s:e], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
            with torch.backends.cudnn.flags(enabled=False):
                out, (h, c) = model(chunk, (h, c))
            h, c = h.contiguous(), c.contiguous()
            preds.extend(out.squeeze(0).cpu().numpy())

    preds = np.array(preds)
    gts = labels[: len(preds)]
    return np.mean((preds - gts)**2), preds, gts

if __name__ == "__main__":
    # Lade Validierungsdaten
    _, df_vals, _, _, _ = load_data()
    name = "MGFarm_18650_C05"
    df_full = df_vals[name]
    # Nur erste 10 %
    n = int(len(df_full) * 0.1)
    df_test = df_full.iloc[:n].reset_index(drop=True)

    # Modell aufsetzen und Gewichte sicher laden
    model = build_model()
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    ))
    model.to(device).eval()

    # Vorhersage und Metriken
    mse, preds, true = evaluate_onechunk_seq2seq(model, df_test, device)
    mae = np.mean(np.abs(preds - true))

    # Plot
    out_dir = Path("test_run"); out_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(range(n), true, label="true SOC")
    plt.plot(range(n), preds, '--', label=f"pred SOC (MAE={mae:.4f})")
    plt.xlabel("Timestep"); plt.ylabel("SOC")
    plt.title(f"Test on {name} (10% data)")
    plt.legend(loc="best"); plt.grid()
    plot_path = out_dir / f"test_{name}.png"
    plt.savefig(plot_path)
    print(f"[INFO] Plot gespeichert unter: {plot_path}")
    print(f"[INFO] MSE={mse:.4f}, MAE={mae:.4f}")
    plt.show()        # Anzeige im Terminal/Notebook
    # plt.close()     # ← optional entfernen oder auskommentieren