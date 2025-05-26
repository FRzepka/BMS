import os
import math
import csv
import torch
import optuna
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 1) Konstanten
BATCH_SIZE = 1024
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# 2) Datenladen & Vorverarbeitung (unverändert)
def load_cell_data(data_dir: Path):
    dfs = {}
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("MGFarm_18650_"):
            p = folder / "df.parquet"
            if p.exists():
                dfs[folder.name] = pd.read_parquet(p)
            else:
                print(f"Warning: {p} fehlt")
    return dfs

def load_data(base_path: str = "/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes"):
    base = Path(base_path)
    cells = load_cell_data(base)
    names = sorted(cells.keys())
    train_cells, val_cell = names[:2], names[2]

    feats = ["Voltage[V]", "Current[A]"]
    train_dfs = {}
    for n in train_cells:
        df = cells[n].copy()
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        train_dfs[n] = df
    scaler = MinMaxScaler((0,1)).fit(pd.concat(train_dfs.values(), ignore_index=True)[feats])
    train_scaled = {n: df.assign(**{c: scaler.transform(df[feats])[:, i]
                                    for i,c in enumerate(feats)}) 
                    for n, df in train_dfs.items()}

    df3 = cells[val_cell].copy()
    df3['timestamp'] = pd.to_datetime(df3['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    L = len(df3); i1, i2 = int(L*0.2), int(L*0.4)
    df_val = df3.iloc[:i1].copy(); df_test = df3.iloc[i1:i2].copy()
    df_val[feats] = scaler.transform(df_val[feats]); df_test[feats] = scaler.transform(df_test[feats])

    return train_scaled, df_val, df_test

# 3) Dataset & Hilfsfunktionen (unverändert)
class CellDataset(Dataset):
    def __init__(self, df, batch_size=BATCH_SIZE):
        self.data = df[["Voltage[V]","Current[A]"]].values
        self.labels = df["SOC_ZHU"].values
        self.batch_size = batch_size
        self.n_batches = max(1, len(self.data)//batch_size)
    def __len__(self):
        return self.n_batches
    def __getitem__(self, idx):
        s = idx*self.batch_size
        e = min(s+self.batch_size, len(self.data))
        return (torch.from_numpy(self.data[s:e]).float(),
                torch.from_numpy(self.labels[s:e]).float())

def init_weights(m):
    if isinstance(m, nn.LSTM):
        for n,p in m.named_parameters():
            if 'weight' in n: nn.init.xavier_uniform_(p)
            elif 'bias' in n: nn.init.constant_(p, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight); nn.init.constant_(m.bias, 0)

def build_model(input_size=2, hidden_size=64, num_layers=1, dropout=0.2, mlp_hidden=16):
    class SOCModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm  = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.0)
            self.mlp   = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1),
                nn.Sigmoid()
            )
        def forward(self, x, hidden):
            out, hidden = self.lstm(x, hidden)
            b, seq, h = out.size()
            flat = out.contiguous().view(b*seq, h)
            soc  = self.mlp(flat).view(b, seq)
            return soc, hidden
    model = SOCModel().to(DEVICE)
    model.apply(init_weights)
    model.lstm.flatten_parameters()
    return model

def init_hidden(model, batch_size=1):
    h = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size, device=DEVICE)
    return (h, torch.zeros_like(h))

def evaluate_online(model, df, max_norm=None):
    model.eval()
    h, c = init_hidden(model)
    preds, gts = [], []
    with torch.no_grad():
        for v, i in tqdm(zip(df['Voltage[V]'].values, df['Current[A]'].values),
                         total=len(df), desc="Validation", leave=False):
            x = torch.tensor([[v,i]], dtype=torch.float32, device=DEVICE).view(1,1,2)
            (p,), (h, c_) = model(x, (h, c))
            preds.append(p.item()); gts.append(df['SOC_ZHU'].iloc[len(preds)-1])
            h, c = h.detach(), c_.detach()
    return np.mean((np.array(preds)-np.array(gts))**2)

# 4) Optuna-Objective
def objective(trial):
    # Hyperparameter-Raum
    lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd          = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [32,64,128,256])
    num_layers  = trial.suggest_int("num_layers", 1, 3)
    dropout     = trial.suggest_float("dropout", 0.0, 0.5)
    mlp_hidden  = trial.suggest_categorical("mlp_hidden", [8,16,32,64])
    max_norm    = trial.suggest_float("max_norm", 0.5, 2.0)

    # Fortschritts-Logging
    print(f"\n--- Trial {trial.number} starting with params: "
          f"lr={lr:.2e}, wd={wd:.2e}, hidden_size={hidden_size}, "
          f"num_layers={num_layers}, dropout={dropout:.2f}, "
          f"mlp_hidden={mlp_hidden}, max_norm={max_norm:.2f}")

    # Daten & Modell
    train_scaled, df_val, _ = load_data()
    model = build_model(hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        mlp_hidden=mlp_hidden)
    optim     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optim, mode="min", patience=3, factor=0.5)
    criterion = nn.MSELoss()
    scaler    = GradScaler()

    # Training
    for ep in range(1, EPOCHS+1):
        print(f"\n=== Trial {trial.number} Epoch {ep}/{EPOCHS} ===")
        model.train()
        total_loss, steps = 0.0, 0
        for name, df in train_scaled.items():
            ds = CellDataset(df, batch_size=BATCH_SIZE)
            print(f"--> {name}, Batches: {len(ds)}")
            dl = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True)
            h, c = init_hidden(model)
            for x_b, y_b in tqdm(dl, desc=f"{name} Ep{ep}", leave=True):
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                optim.zero_grad()
                with autocast(device_type=DEVICE.type):
                    pred, (h,c_) = model(x_b, (h,c))
                    loss = criterion(pred, y_b)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                clip_grad_norm_(model.parameters(), max_norm=max_norm)
                scaler.step(optim)
                scaler.update()
                h, c = h.detach(), c_.detach()
                total_loss += loss.item(); steps += 1

        # Validation & Pruning
        print(f"--> Trial {trial.number} Epoch {ep} running online validation")
        val_mse = evaluate_online(model, df_val, max_norm)
        trial.report(val_mse, ep)
        scheduler.step(val_mse)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return math.sqrt(val_mse)

# 5) Study starten & Ergebnisse loggen
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=30)

    # Ergebnisse als CSV
    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv", index=False)
    print("Best trial:", study.best_trial.params)
    print("Best RMSE:", study.best_value)
