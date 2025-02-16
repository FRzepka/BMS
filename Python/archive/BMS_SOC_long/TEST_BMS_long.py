import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# Device-Auswahl
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Laden der Daten und des trainierten Modells
data_dir = Path('/home/florianr/MG_Farm/5_Data/MGFarm_18650_Dataframes')

def load_all_cell_data(data_dir: Path):
    """
    Durchsucht das Verzeichnis `data_dir` nach allen Unterordnern,
    die mit 'MGFarm_18650_C' beginnen. In jedem solchen Ordner wird
    die Datei 'df.parquet' eingelesen (falls vorhanden).

    Gibt ein Dict zurück:
      {
         "C01": DataFrame,  # aus Ordner MGFarm_18650_C01
         "C02": DataFrame,  # aus Ordner MGFarm_18650_C02
         ...
      }
    """
    dataframes = {}
    for folder in data_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("MGFarm_18650_C"):
            cell_name = folder.name.replace("MGFarm_18650_", "")
            df_path = folder / 'df.parquet'
            if df_path.exists():
                df = pd.read_parquet(df_path)
                dataframes[cell_name] = df
                print(f"Loaded: {folder.name} -> Key: {cell_name}")
            else:
                print(f"Warning: No df.parquet found in {folder.name}")
    return dataframes

cell_data = load_all_cell_data(data_dir)

# Sicherstellen, dass es mindestens 4 Zellen gibt:
if len(cell_data) < 4:
    raise ValueError(
        f"Es wurden nur {len(cell_data)} Zellen gefunden. "
        "Bitte sicherstellen, dass mindestens 4 Zellen vorhanden sind."
    )

# Genau 4 Zellen auswählen (z.B. alphabetisch sortiert, dann die ersten 4 nehmen):
all_cells = sorted(cell_data.keys())
selected_cells = all_cells[:4]
print("Verwendete Zellen:", selected_cells)

# Reduziere cell_data auf die 4 gewählten Zellen
cell_data_4 = {c: cell_data[c] for c in selected_cells}

test_cell = selected_cells[3]
print(f"Test-Zelle: {test_cell}")

# 2) Laden des trainierten Modells
class LSTMSOCModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x.shape = (batch_size, seq_length, input_size=3)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # letzter Zeitschritt im Output:
        last_out = lstm_out[:, -1, :]  # shape = (batch_size, hidden_size)
        soc_pred = self.fc(last_out)   # shape = (batch_size, 1)
        return soc_pred.squeeze(-1)    # -> (batch_size,)

model = LSTMSOCModel(input_size=3, hidden_size=32, num_layers=1).to(device)
model.load_state_dict(torch.load("lstm_soc_model.pth"))
model.eval()

# 3) Vorbereitung der Testdaten
df_test_full = cell_data_4[test_cell].copy()
n_test_samples = int(len(df_test_full) * 0.01)  # 1% der Daten
if n_test_samples < 1:
    n_test_samples = 1

df_test = df_test_full.iloc[:n_test_samples].copy()
print(f"Test-Daten: {len(df_test)} Zeilen (von {len(df_test_full)})")

features = ["Scaled_Voltage[V]", "Scaled_Current[A]", "Scaled_Temperature[°C]"]
test_array = df_test[features + ["SOC_ZHU"]].values

# 4) Autoregressive Vorhersage
def predict_autoregressive(model, data_array, seq_len=60):
    """
    Autoregressive Vorhersage:
    Nutzt das erste seq_len-Fenster mit echten SOC-Werten,
    ab Schritt seq_len wird SOC[t+1] durch Modell ersetzt.
    """
    model.eval()
    data_clone = data_array.copy()
    preds = np.full(len(data_clone), np.nan)

    with torch.no_grad():
        for i in tqdm(range(seq_len, len(data_clone)), desc="Prediction"):
            input_seq = data_clone[i-seq_len : i]  # (seq_len, 3)
            x_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred_soc = model(x_t).item()
            preds[i] = pred_soc
            # Autoregressiv: SOC-Spalte überschreiben
            data_clone[i, 3] = pred_soc

    return preds

seq_length = 60
preds_test = predict_autoregressive(model, test_array, seq_len=seq_length)

# 5) Visualisierung der Ergebnisse
gt_test = df_test["SOC_ZHU"].values
time_test = df_test["timestamp"]

plt.figure(figsize=(12, 5))
plt.plot(time_test, gt_test, label="Ground Truth SOC", color='k')
plt.plot(time_test, preds_test, label="Predicted SOC (autoregressive)", color='r', alpha=0.7)
plt.title(f"Autoregressive SOC-Vorhersage – Test-Zelle: {test_cell}")
plt.xlabel("Time")
plt.ylabel("SOC")
plt.legend()
plt.tight_layout()
plt.show()

# 6) Ausgabe der Ergebnisse
print(f"Test MSE: {np.mean((preds_test[seq_length:] - gt_test[seq_length:])**2):.4f}")
print("Fertig! Test abgeschlossen.")
