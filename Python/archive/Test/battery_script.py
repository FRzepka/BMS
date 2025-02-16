import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Batterie-Skript mit einer Hauptfunktion main().
----------------------------------------------
1) generate_battery_profiles:
   - Erzeugt Profile mit längeren Lade-/Entladephasen ("Blöcke").
   - SoC startet bei 100 %.
   - Stromblöcke mit zufälliger Dauer (z.B. 10..30 Sek).
   - Erreicht SoC 0 % / 100 %, wird der Block beendet und im nächsten Block
     wird ggf. eine neue Zufallsrichtung gewählt.

2) plot_battery_profile:
   - Zeichnet SoC, Strom und Spannung

3) main():
   - Beispielaufruf

Hinweis für Jupyter Notebook:
-----------------------------
- In einer separaten Notebook-Zelle:
    %matplotlib widget
  (sofern ipympl installiert) für interaktive Plots.
"""

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
    Erzeugt eine definierte Anzahl an Batterie-Profilen (1..num_profiles).
    Wir nutzen "Block-Phasen":
      - Pro Block wird eine konstante Stromstärke (Zufallsbetrag) gewählt.
      - SoC startet bei 100%.
      - Wird SoC=0% oder SoC=100% erreicht, ist der Block beendet, und
        im nächsten Block wird eine neue Zufallsrichtung gewählt.
      - Jeder Block dauert zwischen min_block und max_block Zeitschritten (zufällig).

    Parameter:
    -----------
    num_profiles : int
        Anzahl der zu erzeugenden Profile
    capacity_ah  : float
        Nennkapazität (Ah) (Standard=1.0 Ah)
    duration_s   : int
        Gesamtdauer (Sekunden)
    time_step_s  : int
        Zeitauflösung (Sekunden)
    temp_const   : float
        Konstante Temperatur
    current_max  : float
        Maximaler Strom-Betrag (z.B. 10 A)
    min_block    : int
        Minimale Blocklänge in Zeitschritten
    max_block    : int
        Maximale Blocklänge in Zeitschritten

    Returns:
    --------
    dict: {1..num_profiles: pd.DataFrame(...)}
          DataFrame mit time, current, voltage, temperature, SoC
    """
    # Kapazität in Coulomb
    capacity_coulomb = capacity_ah * 3600.0

    # Zeitvektor
    time_vector = np.arange(0, duration_s + time_step_s, time_step_s)
    n_steps = len(time_vector)

    all_profiles = {}

    for p in range(1, num_profiles + 1):
        # Start SoC = 100%
        soc = 1.0
        times = []
        currents = []
        voltages = []
        temperatures = []
        socs = []

        step_idx = 0

        # So lange, bis alle Zeit-Schritte abgedeckt sind
        while step_idx < n_steps:
            # Länge des nächsten Blocks zufällig festlegen
            block_length = np.random.randint(min_block, max_block + 1)
            # Lade oder Entlade-Richtung?
            direction = np.random.choice([-1, 1])
            # Zufälliger Betrag
            magnitude = np.random.uniform(0, current_max)
            block_current = direction * magnitude

            block_count = 0
            while block_count < block_length and step_idx < n_steps:
                # Zeit
                t = time_vector[step_idx]
                times.append(t)

                # SoC merken
                socs.append(soc)

                # Strom
                currents.append(block_current)

                # OCV (sehr einfach)
                voltage = 3.0 + 1.2 * soc
                voltages.append(voltage)

                # Temperatur
                temperatures.append(temp_const)

                # SoC-Update
                dt = 1  # time_step_s
                soc_new = soc + (block_current * dt) / capacity_coulomb

                # Grenzen abfangen
                if soc_new > 1.0:
                    soc_new = 1.0
                    # Block beenden => wir haben 100% erreicht
                    block_count = block_length
                elif soc_new < 0.0:
                    soc_new = 0.0
                    # Block beenden => wir haben 0% erreicht
                    block_count = block_length

                soc = soc_new

                step_idx += 1
                block_count += 1

        profile_df = pd.DataFrame({
            'time': times,
            'current': currents,
            'voltage': voltages,
            'temperature': temperatures,
            'SoC': socs
        })
        all_profiles[p] = profile_df

    return all_profiles


def plot_battery_profile(df_profile):
    """
    Zeichnet drei Subplots:
    1) SoC (%)
    2) Strom (A)
    3) Spannung (V)
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # SoC in %
    axes[0].plot(df_profile['time'], df_profile['SoC'] * 100, label='SoC (%)', color='blue')
    axes[0].set_ylabel('SoC (%)')
    axes[0].grid(True)
    axes[0].legend(loc='best')

    # Strom
    axes[1].plot(df_profile['time'], df_profile['current'], label='Current (A)', color='red')
    axes[1].set_ylabel('Current (A)')
    axes[1].grid(True)
    axes[1].legend(loc='best')

    # Spannung
    axes[2].plot(df_profile['time'], df_profile['voltage'], label='Voltage (V)', color='green')
    axes[2].set_ylabel('Voltage (V)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True)
    axes[2].legend(loc='best')
    axes[2].set_xlim([0, df_profile['time'].max()])

    plt.tight_layout()
    plt.show()


def main():
    """
    Hauptfunktion (Beispiel).
    """
    # Wir erzeugen hier 3 Profile für eine 1Ah-Zelle, 300s, max 10A
    profiles = generate_battery_profiles(
        num_profiles=3,
        capacity_ah=1.0,
        duration_s=300,
        time_step_s=1,
        temp_const=25.0,
        current_max=10.0,
        min_block=10,
        max_block=30
    )

    # Wir picken uns Profil Nr. 1 raus
    df_profile1 = profiles[1]
    print('Erste 5 Zeilen von Profil #1:')
    print(df_profile1.head())

    # Plot
    plot_battery_profile(df_profile1)


if __name__ == "__main__":
    # Nur wenn dieses Skript direkt ausgeführt wird,
    # rufen wir unsere Hauptfunktion main() auf.
    main()
