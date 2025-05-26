#!/bin/bash

# Script zum Starten des BMS SOC LSTM Training v1.2.3.7 in einem Screen
# Verwendung: ./start_training_1.2.3.7.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="BMS_SOC_LSTM_stateful_1.2.3.7_Train.py"
SCREEN_NAME="1.2.3.7"

echo "=== BMS SOC LSTM Training v1.2.3.7 Screen Starter ==="
echo "Script Directory: $SCRIPT_DIR"
echo "Training Script: $SCRIPT_NAME"
echo "Screen Name: $SCREEN_NAME"

# Prüfe ob das Python-Script existiert
if [ ! -f "$SCRIPT_DIR/$SCRIPT_NAME" ]; then
    echo "ERROR: Training script $SCRIPT_NAME nicht gefunden in $SCRIPT_DIR"
    exit 1
fi

# Prüfe ob Screen bereits läuft
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "WARNING: Screen '$SCREEN_NAME' läuft bereits!"
    echo "Möchtest du den bestehenden Screen beenden und neu starten? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Beende bestehenden Screen..."
        screen -S "$SCREEN_NAME" -X quit 2>/dev/null
        sleep 2
    else
        echo "Abgebrochen. Du kannst mit 'screen -r $SCREEN_NAME' zum bestehenden Screen zurückkehren."
        exit 0
    fi
fi

# Wechsle ins Script-Verzeichnis
cd "$SCRIPT_DIR" || exit 1

echo "Starte Training in Screen '$SCREEN_NAME'..."
echo "Verwende: screen -r $SCREEN_NAME um zum Training zurückzukehren"
echo "Verwende: Strg+A dann D um den Screen zu detachen"

# Starte Screen mit dem Training-Script
screen -dmS "$SCREEN_NAME" bash -c "
    echo '=== BMS SOC LSTM Training v1.2.3.7 gestartet ==='
    echo 'Screen Name: $SCREEN_NAME'
    echo 'Zeitstempel: $(date)'
    echo 'Working Directory: $(pwd)'
    echo '=== Training startet in 3 Sekunden ==='
    sleep 3
    python3 $SCRIPT_NAME
    echo ''
    echo '=== Training beendet um $(date) ==='
    echo 'Drücke Enter um den Screen zu beenden oder Strg+A dann D zum Detachen'
    read
"

sleep 1

# Prüfe ob Screen erfolgreich gestartet wurde
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "✅ Screen '$SCREEN_NAME' erfolgreich gestartet!"
    echo ""
    echo "Nützliche Befehle:"
    echo "  screen -r $SCREEN_NAME    # Zum Training-Screen wechseln"
    echo "  screen -list              # Alle Screens anzeigen"
    echo "  Strg+A dann D             # Screen detachen (im Screen)"
    echo "  screen -S $SCREEN_NAME -X quit  # Screen beenden (außerhalb)"
    echo ""
    echo "Das Training läuft jetzt im Hintergrund."
else
    echo "❌ ERROR: Screen konnte nicht gestartet werden!"
    exit 1
fi
