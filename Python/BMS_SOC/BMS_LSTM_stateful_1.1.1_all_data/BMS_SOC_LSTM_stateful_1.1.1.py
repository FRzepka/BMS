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
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cell_data(data_dir: Path):
    """
    Lädt alle Unterordner MGFarm_18650_* und gibt dict {name: df} zurück.
    """
    dataframes = {}
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("MGFarm_18650_"):
            dfp = folder / "df.parquet"
            if dfp.exists():
                df = pd.read_parquet(dfp)
                dataframes[folder.name] = df
            else:
                print(f"Warning: {dfp} fehlt")
    return dataframes

def load_data():
    # 1) load all cells
    base = Path("/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes")
    cells = load_cell_data(base)
    names = sorted(cells.keys())
    val_cell = names[2]
    # use all cells except the validation cell for training
    train_cells = [n for n in names if n != val_cell]

    # 2) prepare train‐DFs with timestamp
    train_dfs = {}
    for name in train_cells:
        df = cells[name].copy()
        df["timestamp"] = pd.to_datetime(df["Absolute_Time[yyyy-mm-dd hh:mm:ss]"])
        train_dfs[name] = df

    # 3) concat and fit scaler on train features
    feats = ["Voltage[V]","Current[A]"]
    df_all_train = pd.concat(train_dfs.values(), ignore_index=True)
    scaler = StandardScaler().fit(df_all_train[feats])

    # 4) scale each train DF
    train_scaled = {}
    for name, df in train_dfs.items():
        df2 = df.copy()
        df2[feats] = scaler.transform(df2[feats])
        train_scaled[name] = df2

    # 5) prepare val+test slices from third cell
    df3 = cells[val_cell].copy()
    df3["timestamp"] = pd.to_datetime(df3["Absolute_Time[yyyy-mm-dd hh:mm:ss]"])
    L = len(df3); i1, i2 = int(L*0.4), int(L*0.8)
    df_val = df3.iloc[:i1].copy()
    df_test= df3.iloc[i2:].copy()
    df_val[feats]  = scaler.transform(df_val[feats])
    df_test[feats] = scaler.transform(df_test[feats])

    return train_scaled, df_val, df_test, train_cells, val_cell

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
    def __init__(self, input_size=2, hidden_size=32, num_layers=1, dropout=0.2, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=batch_first, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.fc(out).squeeze(-1)
        return out, hidden

def train_stateful():
    # load & split
    train_scaled, df_val, df_test, train_cells, val_cell = load_data()
    print("Training on cells:", train_cells)
    print("Validating+testing on cell:", val_cell)

    # plot data
    plt.figure(figsize=(10,4))
    # plot only first three training cells for illustration
    for name, df in list(train_scaled.items())[:3]:
        plt.plot(df["timestamp"], df["SOC_ZHU"], label=f"train {name}")
    plt.legend(); plt.title("Train SOC"); plt.tight_layout()
    plt.savefig("train_data_plot.png"); plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(df_val["timestamp"], df_val["SOC_ZHU"], 'C1', label="val")
    plt.legend(); plt.title("Val SOC"); plt.tight_layout()
    plt.savefig("val_data_plot.png"); plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(df_test["timestamp"], df_test["SOC_ZHU"], 'C2', label="test")
    plt.legend(); plt.title("Test SOC"); plt.tight_layout()
    plt.savefig("test_data_plot.png"); plt.close()

    # prepare concatenated train dataset
    df_train_all = pd.concat(train_scaled.values(), ignore_index=True)
    train_ds = StatefulDataset(df_train_all, chunk_size=256)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)

    # model setup
    model = LSTMSOCModel().to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = StepLR(optim, step_size=5, gamma=0.5)
    best = float('inf')

    for ep in range(1, 21):
        model.train()
        h = c = None; total = 0
        for x, y in tqdm(train_dl, desc=f"Epoch {ep}"):
            x, y = x.to(device), y.to(device)
            if h is None:
                h = torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size, device=device)
                c = torch.zeros_like(h)
            else:
                h, c = h.detach(), c.detach()
            optim.zero_grad()
            out, (h, c) = model(x, (h, c))
            loss = criterion(out.view(-1), y.view(-1))
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total += loss.item()
        ml = total / len(train_dl)
        if ml < best:
            best = ml
            torch.save(model.state_dict(), "best_lstm_soc_stateful.pth")
        print(f"Epoch {ep}, loss={ml:.6f}")
        scheduler.step()

def test_stateful():
    # load & split again
    _, df_val, df_test, train_cells, val_cell = load_data()
    model = LSTMSOCModel().to(device)
    model.load_state_dict(torch.load("best_lstm_soc_stateful.pth",
                                     map_location=device,
                                     weights_only=True))
    model.eval()

    # dataset on df_test
    test_ds = StatefulDataset(df_test, chunk_size=256)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    preds, gts = [], []
    h=c=None
    with torch.no_grad():
        for x,y in tqdm(test_dl, desc="Testing"):
            x,y = x.to(device), y.to(device)
            if h is None:
                h = torch.zeros(model.lstm.num_layers,1,model.lstm.hidden_size,device=device)
                c = torch.zeros_like(h)
            else:
                h, c = h.detach(), c.detach()
            out, (h,c) = model(x,(h,c))
            preds.append(out.view(-1).cpu().numpy())
            gts.append(y.view(-1).cpu().numpy())
    preds = np.concatenate(preds); gts = np.concatenate(gts)
    # compute metrics
    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts)**2))
    # full test plot with metrics
    plt.figure(figsize=(10,4))
    plt.plot(df_test["timestamp"].iloc[:len(preds)], gts, 'k-', label="GT")
    plt.plot(df_test["timestamp"].iloc[:len(preds)], preds, 'r-', label="Pred")
    plt.legend(); plt.title("Final Test")
    plt.text(0.01, 0.95, f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}",
             transform=plt.gca().transAxes, verticalalignment='top')
    plt.tight_layout()
    plt.savefig("final_test_plot.png"); plt.close()
    print("Test plot saved to final_test_plot.png")
    # detailed view: first 50000 points
    timestamps = df_test["timestamp"].iloc[:len(preds)]
    zoom_n = 50000
    plt.figure(figsize=(10,4))
    plt.plot(timestamps[:zoom_n], gts[:zoom_n], 'k-', label="GT")
    plt.plot(timestamps[:zoom_n], preds[:zoom_n], 'r-', label="Pred")
    plt.legend(); plt.title("Test Detail Start (first 50000)")
    plt.tight_layout()
    plt.savefig("final_test_plot_zoom_start.png"); plt.close()
    print("Zoom start plot saved to final_test_plot_zoom_start.png")
    # detailed view: last 50000 points
    plt.figure(figsize=(10,4))
    plt.plot(timestamps[-zoom_n:], gts[-zoom_n:], 'k-', label="GT")
    plt.plot(timestamps[-zoom_n:], preds[-zoom_n:], 'r-', label="Pred")
    plt.legend(); plt.title("Test Detail End (last 50000)")
    plt.tight_layout()
    plt.savefig("final_test_plot_zoom_end.png"); plt.close()
    print("Zoom end plot saved to final_test_plot_zoom_end.png")

if __name__=="__main__":
    train_stateful()
    test_stateful()