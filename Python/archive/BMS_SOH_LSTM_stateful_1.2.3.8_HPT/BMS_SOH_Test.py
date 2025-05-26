import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MaxAbsScaler
import math

# Konstanten
BASE_DATA = "/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes"
MODEL_PATH = "/home/florianr/MG_Farm/6_Scripts/BMS/Python/BMS_SOC/" \
             "BMS_SOH_LSTM_stateful_1.2.3.8_HPT/" \
             "trial_00_hs64_dr0.2040_lr1e-05/best_model.pth"
SEQ_CHUNK_SIZE = 4096

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(base_path: str = BASE_DATA):
    # lade alle Zellen mit nur den benötigten Spalten
    base = Path(base_path)
    needed = ["Absolute_Time[yyyy-mm-dd hh:mm:ss]",
              "Voltage[V]", "Current[A]", "Q_m", "SOH_ZHU"]
    cells = {}
    for f in sorted(base.iterdir()):
        if f.is_dir() and f.name.startswith("MGFarm_18650_"):
            dfp = f/"df.parquet"
            if dfp.exists():
                df = pd.read_parquet(dfp, columns=needed)
                df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
                cells[f.name] = df
    # Skaler über alle Zellen fitten
    feats = ["Voltage[V]", "Current[A]", "Q_m"]
    all_feats = pd.concat([df[feats] for df in cells.values()], ignore_index=True)
    scaler = MaxAbsScaler().fit(all_feats)
    # Test-Zelle (C07) vollständig skalieren
    df_test = cells["MGFarm_18650_C07"].copy()
    df_test[feats] = scaler.transform(df_test[feats])
    return df_test

def build_model(input_size=3, hidden_size=64,
                num_layers=1, dropout=0.2040, mlp_hidden=32):
    class SOHModel(nn.Module):
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
            self.lstm.flatten_parameters()
            out, hidden = self.lstm(x, hidden)
            b, seq, h = out.size()
            flat = out.contiguous().view(b*seq, h)
            soc = self.mlp(flat).view(b, seq)
            return soc, hidden
    model = SOHModel().to(device)
    return model

def init_hidden(model, batch_size=1):
    h = torch.zeros(model.lstm.num_layers, batch_size,
                    model.lstm.hidden_size, device=device)
    c = torch.zeros_like(h)
    return (h, c)

def evaluate_onechunk_seq2seq(model, df, device):
    seq   = df[["Voltage[V]","Current[A]","Q_m"]].values
    labels= df["SOH_ZHU"].values
    chunk = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
    h, c  = init_hidden(model)
    with torch.no_grad():
        model.lstm.flatten_parameters()
        out, _ = model(chunk, (h, c))
    preds = out.squeeze(0).cpu().numpy()
    mse   = np.mean((preds - labels)**2)
    return mse, preds, labels

if __name__ == "__main__":
    # Daten einlesen & skalieren
    df_test = load_data()
    # Modell laden
    model = build_model()
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    # Vorhersage & Metriken
    mse, preds, labels = evaluate_onechunk_seq2seq(model, df_test, device)
    rmse = math.sqrt(mse)
    print(f"Test MSE: {mse:.6f}, RMSE: {rmse:.6f}")
    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(df_test['timestamp'], labels, 'k-', label="True SOH")
    plt.plot(df_test['timestamp'], preds,  'r-', label="Predicted SOH")
    plt.xlabel("Zeit")
    plt.ylabel("SOH_ZHU")
    plt.title(f"SOH-Vorhersage (RMSE={rmse:.4f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("SOH_test_plot.png")
    plt.show()
