#!/bin/bash
set -x

# Array mit Sitzungskonfigurationen: Sitzungsname;Skriptverzeichnis;Skriptname
scripts=(
  # "HPT_NBeats_SOC_extended;/home/florianr/MG_Farm/6_Scripts/NBEATS_SOC_SOH/Python/NBEATS_SOC;1.HPT_NBeats_SOC_extended.py"
  # "HPT_NBeats_SOC;/home/florianr/MG_Farm/6_Scripts/NBEATS_SOC_SOH/Python/NBEATS_SOC;1.HPT_NBeats_SOC.py"
  "Training_NBEATS_SOC;/home/florianr/MG_Farm/6_Scripts/NBEATS_SOC_SOH/Python/NBEATS_SOC;2.Training_NBEATS_SOC.py"
  "Training_NBEATS_SOC_extended;/home/florianr/MG_Farm/6_Scripts/NBEATS_SOC_SOH/Python/NBEATS_SOC;2.Training_NBEATS_SOC_extended.py"
  # "HPT_NBeats_SOH;/home/florianr/MG_Farm/6_Scripts/NBEATS_SOC_SOH/Python/NBEATS_SOH;1.HPT_NBeats_SOH.py"
  # "HPT_NBeats_SOH_cv;/home/florianr/MG_Farm/6_Scripts/NBEATS_SOC_SOH/Python/NBEATS_SOH;1.HPT_NBeats_SOH_cv.py"
  #"Training_NBEATS_SOH;/home/florianr/MG_Farm/6_Scripts/NBEATS_SOC_SOH/Python/NBEATS_SOH;2.Training_NBEATS_SOH.py"
  #"Training_NBEATS_SOH_cv;/home/florianr/MG_Farm/6_Scripts/NBEATS_SOC_SOH/Python/NBEATS_SOH;2.Training_NBEATS_SOH_cv.py"
)

for script_info in "${scripts[@]}"
do
  IFS=';' read -r session_name script_dir script_name <<< "$script_info"

  # Prüfen, ob die Screen-Session bereits läuft
  if screen -list | grep -q "$session_name"; then
    echo "Screen-Session '$session_name' läuft bereits."
  else
    # Neue Screen-Session starten und das Skript ausführen
    screen -dmS "$session_name" bash -c "
      conda activate ml
      cd '$script_dir'
      python '$script_name'
      exec bash"
    echo "Screen-Session '$session_name' gestartet."
  fi
done