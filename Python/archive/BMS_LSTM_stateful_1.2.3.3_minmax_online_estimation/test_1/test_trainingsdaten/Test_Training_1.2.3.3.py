import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MaxAbsScaler
import math

# Gerät definieren und cuDNN optimieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

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
    scaler = MaxAbsScaler().fit(df_all_train[feats])

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
    i1, i2 = int(L*0.2), int(L*0.4)
    df_val = df3.iloc[:i1].copy()
    df_test = df3.iloc[i1:i2].copy()
    df_val[feats] = scaler.transform(df_val[feats])
    df_test[feats] = scaler.transform(df_test[feats])

    return train_scaled, df_val, df_test, train_cells, val_cell

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

# Modell: LSTM + Dropout + MLP-Head
def build_model(input_size=2, hidden_size=64, num_layers=1, dropout=0.2, mlp_hidden=16):
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

def test_train_on_training_data(model_path):
    """
    Lade das vortrainierte Modell und evaluiere auf den Trainingszellen.
    """
    print(f"Loading model from {model_path}")
    train_scaled, _, _, train_cells, _ = load_data()
    model = build_model()
    # avoid future warning
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded. Evaluating on cells: {train_cells}")

    for name, df in train_scaled.items():
        seq = df[["Voltage[V]", "Current[A]"]].values
        print(f"\n--- Cell: {name} | Samples: {len(seq)} ---")
        gts = df["SOC_ZHU"].values
        timestamps = df["timestamp"].values

        feats = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
        h0, c0 = init_hidden(model, batch_size=1, device=device)
        with torch.backends.cudnn.flags(enabled=False), torch.no_grad():
            model.lstm.flatten_parameters()
            preds, _ = model(feats, (h0, c0))
        preds = preds.squeeze(0).cpu().numpy()

        rmse = math.sqrt(((preds - gts) ** 2).mean())
        mae = np.mean(np.abs(preds - gts))
        print(f"Results -> MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # gesamte Kurve
        out_full = f"train_{name}_seq2seq_plot.png"
        print(f"Saving full curve plot to {out_full}")
        plt.figure(figsize=(10,4))
        plt.plot(timestamps, gts, 'k-', label="GT")
        plt.plot(timestamps, preds, 'r-', label="Pred")
        plt.title(f"Train {name} Seq2Seq")
        plt.legend()
        plt.annotate(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}", xy=(0.01,0.95), xycoords='axes fraction', va='top')
        plt.tight_layout(); plt.savefig(out_full); plt.close()

        # Zoom vorne und hinten
        zoom_n = min(50000, len(preds))
        for seg_name, seg in [("start", slice(0, zoom_n)), ("end", slice(-zoom_n, None))]:
            plt.figure(figsize=(10,4))
            plt.plot(timestamps[seg], gts[seg], 'k-', label="GT")
            # sliced preds verwenden, nicht das vollständige Array
            plt.plot(timestamps[seg], preds[seg], 'r-', label="Pred")
            plt.title(f"Train {name} Zoom {seg_name.capitalize()}")
            plt.legend()
            plt.tight_layout(); plt.savefig(f"train_zoom_{seg_name}_{name}.png"); plt.close()

if __name__ == "__main__":
    model_path = "/home/florianr/MG_Farm/6_Scripts/BMS/Python/BMS_SOC/BMS_LSTM_stateful_1.2.3.3_minmax_online_estimation/test_1/best_seq2seq_soc.pth"
    test_train_on_training_data(model_path)