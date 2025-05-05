import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

# Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def load_data():
    base = Path("/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes")
    cells = load_cell_data(base)
    names = sorted(cells.keys())
    train_cells, val_cell = names[:2], names[2]

    feats = ["Voltage[V]", "Current[A]"]
    # 1) Trainingsdaten
    train_dfs = {name: cells[name].assign(
                    timestamp=pd.to_datetime(cells[name]['Absolute_Time[yyyy-mm-dd hh:mm:ss]']))
                 for name in train_cells}

    # 2) Skalar
    df_all_train = pd.concat(train_dfs.values(), ignore_index=True)
    scaler = StandardScaler().fit(df_all_train[feats])

    # 3) Skalierte Trainings-DFs
    train_scaled = {name: df.assign(**{f: scaler.transform(df[feats])[:, i]
                                       for i, f in enumerate(feats)})
                    for name, df in train_dfs.items()}

    # 4) Val/Test Split
    df3 = cells[val_cell].assign(
        timestamp=pd.to_datetime(cells[val_cell]['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    )
    L = len(df3); i1, i2 = int(L*0.4), int(L*0.8)
    df_val = df3.iloc[:i1].copy(); df_test = df3.iloc[i2:].copy()
    df_val.loc[:, feats] = scaler.transform(df_val[feats])
    df_test.loc[:, feats] = scaler.transform(df_test[feats])

    return train_scaled, df_val, df_test, train_cells, val_cell

# Online-Dataset
enabled_timestamps = False
class OnlineDataset(Dataset):
    def __init__(self, df):
        self.x = df[["Voltage[V]","Current[A]"]].values
        self.y = df['SOC_ZHU'].values

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

# Modell mit tieferem LSTM und MLP-Head
def build_model(input_size=2, hidden_size=64, num_layers=2, dropout=0.2, mlp_hidden=32):
    class SOCModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1)
            )

        def forward(self, x, hidden):
            out, hidden = self.lstm(x, hidden)
            last = out[:, -1, :]
            return self.mlp(last).squeeze(-1), hidden

    return SOCModel().to(device)

from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_online(epochs=30, lr=1e-3):
    train_scaled, df_val, df_test, train_cells, val_cell = load_data()
    print("Training auf Zellen:", train_cells)

    # Rohdaten-Plots
    for name, df in train_scaled.items():
        plt.figure(figsize=(10,4))
        plt.plot(df['timestamp'], df['SOC_ZHU'], label=name)
        plt.title(f"Train SOC {name}")
        plt.tight_layout(); plt.savefig(f"train_{name}_plot.png"); plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(df_val['timestamp'], df_val['SOC_ZHU'])
    plt.title("Val SOC"); plt.tight_layout(); plt.savefig("val_data_plot.png"); plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(df_test['timestamp'], df_test['SOC_ZHU'])
    plt.title("Test SOC"); plt.tight_layout(); plt.savefig("test_data_plot.png"); plt.close()

    # Dataset & Loader
    df_train_all = pd.concat(train_scaled.values(), ignore_index=True)
    train_ds = OnlineDataset(df_train_all)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)

    model = build_model()
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optim, factor=0.5, patience=5)

    best_loss = float('inf')
    for ep in range(1, epochs+1):
        model.train()
        h = torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)
        total_loss = 0.0

        for x_t, y_t in tqdm(train_dl, desc=f"Epoch {ep}"):
            x = x_t.unsqueeze(0).to(device)  # (1,1,2)
            y = y_t.unsqueeze(0).to(device)  # (1,)
            optim.zero_grad()
            pred, (h, c) = model(x, (h, c))
            pred = pred.view(-1)  # (1,)
            target = y.view(-1)
            loss = criterion(pred, target)
            loss.backward(); optim.step()
            total_loss += loss.item()
            h, c = h.detach(), c.detach()

        avg_loss = total_loss / len(train_dl)
        scheduler.step(avg_loss)
        print(f"Epoch {ep}, Loss: {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_online_soc.pth")


def test_online():
    _, df_val, df_test, train_cells, val_cell = load_data()
    print("Test auf Zelle:", val_cell)

    test_ds = OnlineDataset(df_test)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = build_model()
    model.load_state_dict(torch.load("best_online_soc.pth", map_location=device))
    model.eval()

    preds, gts = [], []
    h = torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size, device=device)
    c = torch.zeros_like(h)

    with torch.no_grad():
        for x_t, y_t in tqdm(test_dl, desc="Testing"):
            x = x_t.unsqueeze(0).to(device)
            pred, (h, c) = model(x, (h, c))
            preds.append(pred.item()); gts.append(y_t.item())

    preds = np.array(preds); gts = np.array(gts)
    timestamps = df_test['timestamp'].values
    mae = np.mean(np.abs(preds - gts)); rmse = np.sqrt(np.mean((preds - gts)**2))
    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # Gesamt-Plot
    plt.figure(figsize=(10,4))
    plt.plot(timestamps, gts, 'k-', label="GT")
    plt.plot(timestamps, preds, 'r-')
    plt.title("Online Final Test")
    plt.annotate(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}", xy=(0.01,0.95), xycoords='axes fraction', va='top')
    plt.tight_layout(); plt.savefig("final_online_plot.png"); plt.close()

    # Zoom-Plots
    zoom_n = min(50000, len(preds))
    for name, seg in [("Start", slice(0, zoom_n)), ("End", slice(-zoom_n, None))]:
        plt.figure(figsize=(10,4))
        plt.plot(timestamps[seg], gts[seg], 'k-')
        plt.plot(timestamps[seg], preds[seg], 'r-')
        plt.title(f"Zoom {name} (erste/letzte {zoom_n})"); plt.tight_layout()
        plt.savefig(f"zoom_{name.lower()}_online_plot.png"); plt.close()

if __name__ == "__main__":
    train_online()
    test_online()
