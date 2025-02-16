import os
import logging
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["PL_DISABLE_ACCELERATOR_CHECK"] = "1"
# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

# Configure logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import NormalDistributionLoss

import battery_script

# Beschränke die Anzahl der CPU-Threads
NUM_THREADS = 4
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(NUM_THREADS)

def main():
    # 1) Generiere Batterieprofile (1..15)
    profiles = battery_script.generate_battery_profiles(
        num_profiles=15,
        capacity_ah=1.0,
        duration_s=6000,
        time_step_s=1,
        temp_const=25.0,
        current_max=10.0,
        min_block=10,
        max_block=30
    )

    plot_folder = "plots"
    os.makedirs(plot_folder, exist_ok=True)

    # 2) Zusammenführen in ein DataFrame mit Spalten time_idx, series, voltage, current, temperature, soc
    frames = []
    for i, df in profiles.items():
        df_mod = df.copy()
        df_mod["time_idx"] = np.arange(len(df_mod))
        df_mod["series"] = f"battery_{i}"
        df_mod.rename(columns={"SoC": "soc"}, inplace=True)
        frames.append(df_mod)
    all_data = pd.concat(frames, ignore_index=True)

    # Plot exemplarisch battery_1 mit battery_script vor dem Training
    battery_1_data = all_data[all_data.series == "battery_1"].copy()
    battery_1_data.rename(columns={"soc": "SoC"}, inplace=True)
    battery_script.plot_battery_profile(battery_1_data)
    plt.savefig(os.path.join(plot_folder, "battery_1_DeepAR.png"))
    plt.close()

    # 3) Kategoriale Kodierung
    categorical_encoders = {"series": NaNLabelEncoder().fit(all_data.series)}

    # 4) Train-/Val-/Test-Split
    train_data = all_data[all_data.series.isin([f"battery_{i}" for i in range(1, 14)])]
    val_data = all_data[all_data.series == "battery_14"]
    test_data = all_data[all_data.series == "battery_15"]

    max_encoder_length = 60
    max_prediction_length = 5

    # 5) TimeSeriesDataSet für Pytorch Forecasting
    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="soc",
        group_ids=["series"],
        static_categoricals=["series"],
        time_varying_known_reals=["voltage", "current", "temperature"],
        time_varying_unknown_reals=["soc"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        categorical_encoders=categorical_encoders,
        allow_missing_timesteps=True
    )
    validation = TimeSeriesDataSet.from_dataset(training, val_data, predict=True)
    test = TimeSeriesDataSet.from_dataset(training, test_data, predict=True)

    batch_size = 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # 6) Training
    pl.seed_everything(42)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    
    # Überprüfe, ob CUDA verfügbar ist
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("Using CPU.")
    
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu" if cuda_available else "cpu",
        devices=1,  # Setze devices immer auf 1
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=False,
        enable_checkpointing=False,
        detect_anomaly=False,
    )

    model = DeepAR.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=64,
        rnn_layers=3,
        dropout=0.2,
        loss=NormalDistributionLoss()
    )
    
    if cuda_available:
        model = model.cuda()  # Übertrage das Modell explizit auf die GPU
        print("Model has been moved to GPU.")
    else:
        print("Model is using CPU.")
    
    print("Starting training...")
    trainer.fit(model, train_dataloader, val_dataloader)
    print("Training completed.")

    # 7) Autoregressive Prediction Loop
    test_data_autoreg = test_data.copy()
    predictions_autoreg = []
    prediction_times_autoreg = []
    current_window = test_data_autoreg.iloc[:max_encoder_length + max_prediction_length].copy()

    for t in range(max_encoder_length + max_prediction_length, len(test_data_autoreg)):
        current_dataset = TimeSeriesDataSet.from_dataset(
            training,
            current_window,
            predict=True,
            stop_randomization=True,
            max_encoder_length=max_encoder_length,
            min_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            min_prediction_length=max_prediction_length
        )
        current_dataloader = current_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

        with torch.no_grad():
            ml_pred = model.predict(current_dataloader)[0].cpu().numpy()[0]
        final_pred = np.clip(ml_pred, 0, 1)

        predictions_autoreg.append(final_pred)
        prediction_times_autoreg.append(test_data_autoreg.iloc[t]["time_idx"])

        new_row = test_data_autoreg.iloc[t].copy()
        new_row["soc"] = final_pred
        current_window = pd.concat([current_window.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)

    # 8) Fehleranalyse
    error = np.abs(
        np.array(predictions_autoreg)
        - test_data.soc.iloc[max_encoder_length + max_prediction_length:].values
    )
    mae = np.mean(error)
    print(f"Mean Absolute Error: {mae:.4f}")

    # 9) Plot
    plt.figure(figsize=(15, 15))
    plt.subplot(4, 1, 1)
    train_example = train_data[train_data.series == "battery_1"]
    plt.plot(train_example.time_idx, train_example.soc)
    plt.title("Training Example (Battery 1)")
    plt.ylabel("SOC")

    plt.subplot(4, 1, 2)
    plt.plot(val_data.time_idx, val_data.soc)
    plt.title("Validation Data (Battery 14)")
    plt.ylabel("SOC")

    plt.subplot(4, 1, 3)
    plt.plot(test_data.time_idx, test_data.soc)
    plt.title("Test Data Ground Truth (Battery 15)")
    plt.ylabel("SOC")

    plt.subplot(4, 1, 4)
    plt.plot(prediction_times_autoreg, predictions_autoreg, "r--", label="Predictions")
    plt.plot(test_data.time_idx, test_data.soc, "b-", label="Ground Truth")
    plt.title("Autoregressive Predictions vs Ground Truth (Nur Modell)")
    plt.ylabel("SOC")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, "train_val_test_socs_DeepAR.png"))
    plt.show()

if __name__ == "__main__":
    main()