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
    dataframes = {}
    folder = data_dir / "MGFarm_18650_C01"
    if folder.exists() and folder.is_dir():
        df_path = folder / "df.parquet"
        if df_path.exists():
            dataframes["C01"] = pd.read_parquet(df_path)
        else:
            raise FileNotFoundError(f"{df_path} not found")
    else:
        raise FileNotFoundError(f"{folder} not found")
    return dataframes

def load_data():
    data_dir = Path("/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes")
    df_full = load_cell_data(data_dir)[sorted(load_cell_data(data_dir).keys())[0]]
    # Timestamp
    df_full["timestamp"] = pd.to_datetime(df_full["Absolute_Time[yyyy-mm-dd hh:mm:ss]"])
    # splits
    n = len(df_full)
    t1 = int(n*0.4); t2 = int(n*0.8)
    df_train = df_full.iloc[:t1].copy()
    df_val   = df_full.iloc[t1:t2].copy()
    df_test  = df_full.iloc[t2:].copy().reset_index(drop=True)
    # scale
    cols = ["Voltage[V]","Current[A]"]
    scaler = StandardScaler().fit(df_train[cols])
    for df in (df_train, df_val, df_test):
        df[cols] = scaler.transform(df[cols])
    return df_train, df_val, df_test

class StatefulDataset(Dataset):
    def __init__(self, df, chunk_size=256):
        self.x = df[["Voltage[V]","Current[A]"]].values
        self.y = df["SOC_ZHU"].values
        self.N = len(self.x) // chunk_size
        self.cs = chunk_size
    def __len__(self):
        return self.N
    def __getitem__(self, i):
        s, e = i*self.cs, (i+1)*self.cs
        return (torch.tensor(self.x[s:e],dtype=torch.float32),
                torch.tensor(self.y[s:e],dtype=torch.float32))

class LSTMSOCModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size,1)
    def forward(self, x, hc=None):
        if hc is None:
            out, hc = self.lstm(x)
        else:
            out, hc = self.lstm(x, hc)
        return self.fc(out).squeeze(-1), hc

def test_only():
    # load data & model
    _, _, df_test = load_data()
    ds = StatefulDataset(df_test, chunk_size=256)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    model = LSTMSOCModel().to(device)
    model.load_state_dict(torch.load("best_lstm_soc_stateful.pth", map_location=device))
    model.eval()

    # run prediction
    preds, gts = [], []
    h, c = None, None
    for x, y in tqdm(dl, desc="Inferencing"):
        x, y = x.to(device), y.to(device)
        if h is None:
            h = torch.zeros(model.lstm.num_layers, x.size(0), model.lstm.hidden_size, device=device)
            c = torch.zeros(model.lstm.num_layers, x.size(0), model.lstm.hidden_size, device=device)
        else:
            h, c = h.detach(), c.detach()
        out, (h, c) = model(x, (h,c))
        preds.append(out.view(-1).detach().cpu().numpy())
        gts.append(y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    gts   = np.concatenate(gts)

    # trim df_test auf gleiche Länge
    total = len(preds)
    df_t = df_test.iloc[:total]

    # 1) Full Plot
    plt.figure(figsize=(10,4))
    plt.plot(df_t["timestamp"], gts, 'k-', label="GT SOC")
    plt.plot(df_t["timestamp"], preds, 'r-', label="Pred SOC")
    plt.legend(); plt.title("Full Test"); plt.tight_layout()
    plt.savefig("full_test_plot.png"); plt.close()

    # 2) Detail Start (erste 500 Punkte)
    N=50000
    plt.figure(figsize=(8,4))
    plt.plot(df_t["timestamp"][:N], gts[:N], 'k-', label="GT")
    plt.plot(df_t["timestamp"][:N], preds[:N], 'r-', label="Pred")
    plt.legend(); plt.title("Detail Start"); plt.tight_layout()
    plt.savefig("detail_start.png"); plt.close()

    # 3) Detail End (letzte 500 Punkte)
    plt.figure(figsize=(8,4))
    plt.plot(df_t["timestamp"][-N:], gts[-N:], 'k-', label="GT")
    plt.plot(df_t["timestamp"][-N:], preds[-N:], 'r-', label="Pred")
    plt.legend(); plt.title("Detail End"); plt.tight_layout()
    plt.savefig("detail_end.png"); plt.close()

    print("Plots saved: full_test_plot.png, detail_start.png, detail_end.png")

if __name__=="__main__":
    test_only()