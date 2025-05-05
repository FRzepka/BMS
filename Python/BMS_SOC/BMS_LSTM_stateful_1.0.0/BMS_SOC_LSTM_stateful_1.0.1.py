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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cell_data(data_dir: Path):
    """
    Lädt df.parquet aus /MGFarm_18650_C01 und gibt ein dict {cell_name: dataframe} zurück.
    """
    dataframes = {}
    folder = data_dir / "MGFarm_18650_C01"
    if folder.exists() and folder.is_dir():
        df_path = folder / 'df.parquet'
        if df_path.exists():
            df = pd.read_parquet(df_path)
            dataframes["C01"] = df
        else:
            print(f"Warning: No df.parquet found in {folder.name}")
    else:
        print("Warning: Folder MGFarm_18650_C01 not found")
    return dataframes

def load_data():
    data_dir = Path("/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes")
    cell_data = load_cell_data(data_dir)
    cell_name = sorted(cell_data.keys())[0]
    df_full = cell_data[cell_name]

    # ...existing code for selecting smaller portion, time-based splits...
    df_small = df_full.copy()
    df_small['timestamp'] = pd.to_datetime(df_small['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    len_small = len(df_small)
    train_end = int(len_small * 0.4)
    val_end   = int(len_small * 0.8)
    df_train  = df_small.iloc[:train_end].copy()
    df_val    = df_small.iloc[train_end:val_end].copy()
    df_test   = df_small.iloc[val_end:].copy()

    # Skalierung (Voltage, Current)
    features_to_scale = ['Voltage[V]', 'Current[A]']
    scaler = StandardScaler()
    scaler.fit(df_train[features_to_scale])
    df_train_scaled = df_train.copy()
    df_val_scaled   = df_val.copy()
    df_test_scaled  = df_test.copy()
    df_train_scaled[features_to_scale] = scaler.transform(df_train_scaled[features_to_scale])
    df_val_scaled[features_to_scale]   = scaler.transform(df_val_scaled[features_to_scale])
    df_test_scaled[features_to_scale]  = scaler.transform(df_test_scaled[features_to_scale])

    return df_train_scaled, df_val_scaled, df_test_scaled, df_train, df_val, df_test

class StatefulDataset(Dataset):
    """
    Returns consecutive chunks for stateful training/testing.
    Each item is (batch=1, chunk_size, input_dim), (batch=1, chunk_size)
    """
    def __init__(self, df_scaled, chunk_size=256):
        self.x = df_scaled[['Voltage[V]', 'Current[A]']].values
        self.y = df_scaled['SOC_ZHU'].values
        self.chunk_size = chunk_size
        self.num_chunks = (len(self.x) // self.chunk_size)

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = start + self.chunk_size
        x_chunk = torch.tensor(self.x[start:end], dtype=torch.float32)
        y_chunk = torch.tensor(self.y[start:end], dtype=torch.float32)
        return x_chunk, y_chunk

class LSTMSOCModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)
        out = self.fc(out).squeeze(-1)
        return out, hidden

def train_stateful():
    print("Starting train_stateful...")
    df_train_scaled, df_val_scaled, df_test_scaled, df_train, df_val, df_test = load_data()
    train_ds = StatefulDataset(df_train_scaled, chunk_size=256)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)
    model = LSTMSOCModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float('inf')
    for epoch in range(5):
        print(f"Epoch {epoch+1}/5")
        model.train()
        h, c = None, None
        total_loss = 0
        for x, y in tqdm(train_dl, desc="Training Batches"):
            x, y = x.to(device), y.to(device)
            if h is None:  # initialize hidden states for batch_size=1
                h = torch.zeros(model.lstm.num_layers, x.size(0), model.lstm.hidden_size, device=device)
                c = torch.zeros(model.lstm.num_layers, x.size(0), model.lstm.hidden_size, device=device)
            else:
                h = h.detach()
                c = c.detach()

            optimizer.zero_grad()
            out, (h, c) = model(x, (h, c))
            loss = criterion(out.view(-1), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        mean_loss = total_loss / len(train_dl)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), "best_lstm_soc_stateful.pth")
        print(f"Done epoch {epoch+1}, partial train_loss={mean_loss:.6f}")

def test_stateful():
    print("Starting test_stateful...")
    df_train_scaled, df_val_scaled, df_test_scaled, df_train, df_val, df_test = load_data()
    model = LSTMSOCModel().to(device)
    model.load_state_dict(torch.load("best_lstm_soc_stateful.pth", map_location=device))
    model.eval()

    # Build test dataset in consecutive chunks
    test_ds = StatefulDataset(df_test_scaled, chunk_size=256)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Slice df_test to ensure same length as predictions
    total_points = test_ds.num_chunks * test_ds.chunk_size
    df_test = df_test.iloc[:total_points].reset_index(drop=True)

    preds, gts = [], []
    h, c = None, None
    with torch.no_grad():
        for x, y in tqdm(test_dl, desc="Testing Batches"):
            x, y = x.to(device), y.to(device)
            if h is None:
                h = torch.zeros(model.lstm.num_layers, x.size(0), model.lstm.hidden_size, device=device)
                c = torch.zeros(model.lstm.num_layers, x.size(0), model.lstm.hidden_size, device=device)
            else:
                h = h.detach()
                c = c.detach()

            out, (h, c) = model(x, (h, c))
            preds.append(out.view(-1).cpu().numpy())
            gts.append(y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    gts   = np.concatenate(gts)

    # Compare with unscaled test data (df_test)
    t_test = df_test['timestamp'].values
    plt.figure()
    plt.plot(t_test, gts, label='Ground Truth SOC', color='k')
    plt.plot(t_test, preds, label='Predicted SOC', color='r')
    plt.legend()
    plt.savefig("test_stateful_plot.png")
    plt.close()
    print("Test finished, saved plot to test_stateful_plot.png")

if __name__ == "__main__":
    train_stateful()
    test_stateful()