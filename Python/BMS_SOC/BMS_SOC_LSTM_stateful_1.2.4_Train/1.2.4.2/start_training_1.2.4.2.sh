#!/bin/bash

# Script zum Starten des BMS SOC LSTM Training 1.2.4.2 in einem Screen
# Autor: Auto-generiert
# Datum: $(date)

echo "=== Starting BMS SOC LSTM Training 1.2.4.2 ==="

# Screen-Name definieren
SCREEN_NAME="training_1.2.4.2"

# Pfad zum Script
SCRIPT_DIR="/home/florianr/MG_Farm/6_Scripts/BMS/Python/BMS_SOC/BMS_SOC_LSTM_stateful_1.2.4_Train/1.2.4.2"
SCRIPT_NAME="BMS_SOC_LSTM_stateful_1.2.4.2_Train.py"

# Prüfen ob Screen bereits existiert
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "ERROR: Screen '$SCREEN_NAME' existiert bereits!"
    echo "Verwende 'screen -r $SCREEN_NAME' um zu dem Screen zu wechseln"
    echo "Oder 'screen -S $SCREEN_NAME -X quit' um ihn zu beenden"
    exit 1
fi

# Prüfen ob Script existiert
if [ ! -f "$SCRIPT_DIR/$SCRIPT_NAME" ]; then
    echo "ERROR: Script '$SCRIPT_DIR/$SCRIPT_NAME' nicht gefunden!"
    exit 1
fi

echo "Erstelle Screen: $SCREEN_NAME"
echo "Script-Pfad: $SCRIPT_DIR/$SCRIPT_NAME"
echo "Conda Environment: ml1"
echo ""

# Screen erstellen und Commands ausführen
screen -dmS "$SCREEN_NAME" bash -c "
    echo '=== Screen $SCREEN_NAME gestartet ==='
    echo 'Aktiviere Conda Environment ml1...'
    
    # Conda aktivieren
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ml1
    
    echo 'Conda Environment aktiviert: '\$(conda info --envs | grep '*')
    echo ''
    
    # In Script-Verzeichnis wechseln
    cd '$SCRIPT_DIR'
    echo 'Gewechselt zu Verzeichnis: '\$(pwd)
    echo ''
    
    # Python Script starten
    echo 'Starte Training...'
    echo '=================='
    python '$SCRIPT_NAME'
    
    echo ''
    echo '=================='
    echo 'Training beendet!'
    echo 'Drücke Enter um Screen zu beenden...'
    read
"

echo "Screen '$SCREEN_NAME' wurde gestartet!"
echo ""
echo "Nützliche Commands:"
echo "  - Screen anzeigen:     screen -r $SCREEN_NAME"
echo "  - Screen beenden:      screen -S $SCREEN_NAME -X quit"
echo "  - Alle Screens:        screen -list"
echo "  - Vom Screen trennen:  Ctrl+A, dann D"
echo ""
echo "Das Training läuft jetzt im Hintergrund..."
