import os
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss

# Beschränke die Anzahl der CPU-Threads
NUM_THREADS = 4
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(NUM_THREADS)

# Überprüfe, ob CUDA verfügbar ist und setze das Gerät entsprechend
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise RuntimeError("CUDA is not available. Please ensure a GPU is available and properly configured.")

############################################
# Beispiel-Skript:
# Wir generieren Batterie-Profile (synthetisch)
# und trainieren ein TemporalFusionTransformer-Modell
# (TFT) darauf.
############################################

############################################
# 1) Daten generieren
############################################

def generate_battery_profiles(
    num_profiles=15,
    capacity_ah=1.0,
    duration_s=300,
    time_step_s=1,
    temp_const=25.0,
    current_max=10.0,
    min_block=10,
    max_block=30
):
    """
    Erzeugt eine definierte Anzahl an Batterie-Profilen (1..num_profiles) mit:
      - SoC-Start bei 100%
      - Zufällige Lade/Entlade-Blöcke (constant I pro Block, 10..30 s)
      - SoC = 0..1, bei Erreichen der Grenze wird der Block beendet.
    Gibt dict {profil_nr: DataFrame} zurück.

    DataFrame:
      time, current, voltage, temperature, SoC
    """
    capacity_coulomb = capacity_ah * 3600.0
    time_vector = np.arange(0, duration_s + time_step_s, time_step_s)
    n_steps = len(time_vector)

    all_profiles = {}

    for p in range(1, num_profiles + 1):
        soc = 1.0  # Start 100%
        times = []
        currents = []
        voltages = []
        temperatures = []
        socs = []

        step_idx = 0
        while step_idx < n_steps:
            block_length = np.random.randint(min_block, max_block + 1)
            direction = np.random.choice([-1, 1])
            magnitude = np.random.uniform(0, current_max)
            block_current = direction * magnitude

            block_count = 0
            while block_count < block_length and step_idx < n_steps:
                t = time_vector[step_idx]
                times.append(t)
                socs.append(soc)
                currents.append(block_current)
                voltage = 3.0 + 1.2 * soc  # einfache OCV-Kurve
                voltages.append(voltage)
                temperatures.append(temp_const)

                dt = 1
                soc_new = soc + (block_current * dt) / capacity_coulomb

                if soc_new > 1.0:
                    soc_new = 1.0
                    block_count = block_length  # Block beenden
                elif soc_new < 0.0:
                    soc_new = 0.0
                    block_count = block_length

                soc = soc_new
                step_idx += 1
                block_count += 1

        df_profile = pd.DataFrame({
            'time': times,
            'current': currents,
            'voltage': voltages,
            'temperature': temperatures,
            'SoC': socs
        })
        all_profiles[p] = df_profile

    return all_profiles


def plot_battery_profile(df_profile):
    """
    Zeichnet SoC, Strom und Spannung.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(df_profile['time'], df_profile['SoC'] * 100, label='SoC (%)', color='blue')
    axes[0].set_ylabel('SoC (%)')
    axes[0].grid(True)
    axes[0].legend(loc='best')

    axes[1].plot(df_profile['time'], df_profile['current'], label='Current (A)', color='red')
    axes[1].set_ylabel('Current (A)')
    axes[1].grid(True)
    axes[1].legend(loc='best')

    axes[2].plot(df_profile['time'], df_profile['voltage'], label='Voltage (V)', color='green')
    axes[2].set_ylabel('Voltage (V)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True)
    axes[2].legend(loc='best')

    plt.tight_layout()
    plt.show()


############################################
# 2) Hauptfunktion mit TFT-Training
############################################

def main():
    # a) Generiere Daten
    profiles = generate_battery_profiles(
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

    # b) In ein gemeinsames DataFrame
    #    Spalten: [time_idx, series, (voltage, current, temperature, soc)]
    frames = []
    for i, df in profiles.items():
        d = df.copy()
        d["time_idx"] = np.arange(len(d))
        d["series"] = f"battery_{i}"
        # rename SoC => "target", z.B.:
        d.rename(columns={"SoC": "soc"}, inplace=True)
        frames.append(d)

    all_data = pd.concat(frames, ignore_index=True)

    # Plot exemplarisch
    battery_1_data = all_data[all_data.series == "battery_1"].copy()
    battery_1_data.rename(columns={"soc": "SoC"}, inplace=True)
    plot_battery_profile(battery_1_data)
    plt.savefig(os.path.join(plot_folder, "battery_1_TFT.png"))
    plt.close()

    # c) Split in train / val
    #    Hier sehr einfach: erst battery_1..13 => train, 14 => val, 15 => test
    train_data = all_data[all_data.series.isin([f"battery_{i}" for i in range(1, 14)])]
    val_data = all_data[all_data.series == "battery_14"]
    test_data = all_data[all_data.series == "battery_15"]

    # d) max_encoder_length, max_prediction_length
    max_encoder_length = 60  # wie viele Vergangenheitswerte pro Sample
    max_prediction_length = 5 # wie viele Zukunftswerte prädiziert werden

    # e) TimeSeriesDataSet
    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="soc",  # SoC als Vorhersage
        group_ids=["series"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["soc"],
        time_varying_known_reals=["voltage", "current", "temperature"],
        static_categoricals=["series"],
        categorical_encoders={"series": NaNLabelEncoder().fit(all_data.series)},
        add_relative_time_idx=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_data, predict=True)
    test = TimeSeriesDataSet.from_dataset(training, test_data, predict=True)

    batch_size = 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size)
    test_dataloader = test.to_dataloader(train=False, batch_size=batch_size)

    # f) Trainer
    pl.seed_everything(42)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )

    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs", name="battery_TFT")

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator=device,
        devices=1,  # Setze devices immer auf 1
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_logger],
        logger=logger,
        enable_progress_bar=True,
    )

    # Überprüfe, ob CUDA verfügbar ist
    print(f"CUDA available: {torch.cuda.is_available()}")

    # g) Temporal Fusion Transformer
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    # h) Fit
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # i) Test-Predictions
    raw_predictions = tft.predict(
        test_dataloader,
        mode="raw",
        return_x=True
    )

    # Beispiel: Plot die ersten 3 Predictions
    for idx in range(3):
        fig = tft.plot_prediction(
            raw_predictions.x,
            raw_predictions.output,
            idx=idx,
            add_loss_to_title=True,
        )
        plt.savefig(os.path.join(plot_folder, f"prediction_{idx}_TFT.png"))
        plt.close()


def run():
    # Einfacher Wrapper
    main()

if __name__ == "__main__":
    run()
