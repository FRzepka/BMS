import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ordner für Hyperparametertuning-Ergebnisse (HPT)
hpt_folder = Path(__file__).parent / "HPT" if '__file__' in globals() else Path("HPT")
os.makedirs(hpt_folder, exist_ok=True)

def load_multiple_cells(data_dir):
    dataframes = {}
    # Discover matching subfolders
    all_folders = sorted([d for d in data_dir.iterdir() 
                          if d.is_dir() and d.name.startswith("MGFarm_18650_C")])
    # Keep only first 3 folders, but first 2 for training
    selected_folders = all_folders[:3]
    for folder in selected_folders:
        df_path = folder / 'df.parquet'
        if df_path.exists():
            df = pd.read_parquet(df_path)
            if 'Absolute_Time[yyyy-mm-dd hh:mm:ss]' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
            print(f"First row of {folder.name}:\n{df.head(1)}")  # NEW: print first row
            dataframes[folder.name.split("_")[-1]] = df
            print(f"[INFO] Loaded {folder.name}")
        else:
            print(f"[WARN] No df.parquet found in {folder.name}")
    return dataframes

class Chomp1d(nn.Module):
    """Chomps the 'causal padding' in a 1D convolution."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        self.chomp1 = Chomp1d(chomp_size=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        self.chomp2 = Chomp1d(chomp_size=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """
    Simple TCN with a list of TCN blocks.
    - input_size: e.g. 2 for [Voltage, Current]
    - num_channels: list of out_channels per block
    - kernel_size, dropout: hyperparameters
    """
    def __init__(self, input_size=2, num_channels=[32, 32], kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            block = TCNBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=padding,
                dropout=dropout
            )
            layers.append(block)
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size) -> permute -> (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)
        y = self.tcn(x)  # => (batch_size, out_channels, seq_len)
        last_out = y[:, :, -1]  # last time step
        out = self.fc(last_out)
        return out.squeeze(-1)

class SequenceDataset(Dataset):
    """
    For each sample:
     - Input: a window of [Voltage, Current] over seq_len points
     - Label: SOC_ZHU at (t + seq_len)
    """
    def __init__(self, df, seq_len=60):
        self.seq_len = seq_len
        self.features = df[["Voltage[V]", "Current[A]"]].values
        self.labels   = df["SOC_ZHU"].values

    def __len__(self):
        return len(self.labels) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.features[idx : idx + self.seq_len]
        y_val = self.labels[idx + self.seq_len]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

if __name__ == '__main__':
    data_dir = Path('/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes')
    dfs = load_multiple_cells(data_dir)
    
    cell_ids = list(dfs.keys())
    train_cell_ids = cell_ids[:2]
    val_cell_id = cell_ids[2] if len(cell_ids) > 2 else None
    print(f"[INFO] Training on cells: {train_cell_ids}")
    print(f"[INFO] Validation cell: {val_cell_id}")

    # Combine training cells into a single DataFrame for hyperparam tuning
    df_train_combined = pd.DataFrame()
    for cid in train_cell_ids:
        df_temp = dfs[cid].copy()
        df_train_combined = pd.concat([df_train_combined, df_temp], ignore_index=True)

    # NEW: replicate TCM_HPT_1.0 approach, e.g. keep 25% of combined
    sample_size = int(len(df_train_combined) * 0.25)
    df_small = df_train_combined.head(sample_size).copy()
    print(f"[INFO] Using 25% combined => {sample_size} rows")

    # Time-based split (train 40%, val 40%, test 20%)
    df_small['timestamp'] = pd.to_datetime(df_small['timestamp'])
    len_small = len(df_small)
    train_end = int(len_small * 0.4)
    val_end   = int(len_small * 0.8)

    df_train = df_small.iloc[:train_end]
    df_val   = df_small.iloc[train_end:val_end]
    df_test  = df_small.iloc[val_end:]
    print(f"[INFO] Combined split => Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Fit scaler only on train
    features_to_scale = ["Voltage[V]", "Current[A]"]
    scaler = StandardScaler()
    scaler.fit(df_train[features_to_scale])

    df_train_scaled = df_train.copy()
    df_val_scaled   = df_val.copy()
    df_test_scaled  = df_test.copy()
    df_train_scaled[features_to_scale] = scaler.transform(df_train_scaled[features_to_scale])
    df_val_scaled[features_to_scale]   = scaler.transform(df_val_scaled[features_to_scale])
    df_test_scaled[features_to_scale]  = scaler.transform(df_test_scaled[features_to_scale])

    # 1) Possibly scale down or slice for faster tuning
    # ...existing code from TCM_HPT_1.0 hyperparam approach (time-based split, scaling)...

    # 2) Define objective function
    def objective(trial: optuna.Trial):
        # Example hyperparams
        n_ch1 = trial.suggest_int("n_ch1", 16, 128, step=16)
        n_ch2 = trial.suggest_int("n_ch2", 16, 128, step=16)
        seq_length = trial.suggest_int("seq_length", 600, 3600, step=600)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        # ...existing hyperparams if needed...

        # Create train/val DataLoaders similar to TCM_HPT_1.0
        # ...existing code to slice df_train_combined into train_df, val_df...
        # ...existing code to scale features, build SequenceDataset, DataLoader...

        model = TCN(
            input_size=2,
            num_channels=[n_ch1, n_ch2],
            kernel_size=2,
            dropout=dropout
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Simple training loop (adapt as needed)
        n_epochs = 2
        best_val_loss = float('inf')
        best_state = None

        for epoch in range(n_epochs):
            model.train()
            # ...existing train loop code...
            model.eval()
            val_loss = 0.0
            # ...existing validation loop code...
            # e.g., val_loss += partial_loss * batch_size
            # end of validation loop:
            # val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()

            trial.report(best_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Optionally store best_state in a global holder
        # ...existing code...
        return best_val_loss

    # 3) Create and run the study, store trials
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)  # or your desired number of trials

    # 4) Save trials to CSV
    csv_file = hpt_folder / "hpt_trials_2cells.csv"
    # ...existing code to write trial info to csv_file...

    # Retrieve best model state from a global or local holder
    best_model_state = {}  # replaced upon best trial
    best_model_path = hpt_folder / "best_tcn_soc_2cells_model.pth"
    torch.save(best_model_state, best_model_path)
    print(f"[INFO] Best 2-cells model saved at: {best_model_path}")

    # Validation on the third cell, single slice
    if val_cell_id is not None:
        df_val = dfs[val_cell_id].copy()
        start_i, end_i = 0, int(len(df_val)*0.2)  # example slice
        df_slice = df_val.iloc[start_i:end_i].copy()

        # 1) Reuse the same scaler fitted on train earlier
        df_slice_scaled = df_slice.copy()
        df_slice_scaled[features_to_scale] = scaler.transform(df_slice_scaled[features_to_scale])

        # 2) Load the trained TCN:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Replace with actual best channels / hyperparams
        best_tcn = TCN(input_size=2, num_channels=[32, 32], kernel_size=2, dropout=0.2).to(device)
        best_tcn.load_state_dict(torch.load(best_model_path, map_location=device))
        best_tcn.eval()

        # 3) Wrap slice in a SequenceDataset
        seq_length = 60  # or from your best hyperparams
        val_dataset = SequenceDataset(df_slice_scaled, seq_len=seq_length)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=True)

        # 4) Predictions
        predictions = []
        timestamps = []
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                preds = best_tcn(x_batch)
                predictions.append(preds.cpu().numpy())
        predictions = np.concatenate(predictions)

        # Collect the corresponding timestamps for the predicted steps
        full_times = df_slice['timestamp'].values
        timestamps = full_times[seq_length : seq_length + len(predictions)]

        # 5) Plot
        plt.figure(figsize=(10,4))
        plt.plot(timestamps, df_slice['SOC_ZHU'].values[seq_length : seq_length + len(predictions)],
                 label="SOC (GT)", color='black')
        plt.plot(timestamps, predictions, label="SOC (Pred)", linestyle='--', color='red')
        plt.xlabel("Time")
        plt.ylabel("SOC")
        plt.legend()
        plt.show()

    print(f"[INFO] Done. Used cells: {train_cell_ids} for training, {val_cell_id} for validation.")