#!/bin/bash

# Prüfe, ob tmux installiert ist
if ! command -v tmux &> /dev/null; then
    echo "tmux ist nicht installiert. Bitte installiere tmux und versuche es erneut."
    exit 1
fi

# Überprüfe, ob eine tmux-Sitzung namens 'monitor' existiert und töte sie
if tmux has-session -t monitor 2>/dev/null; then
    tmux kill-session -t monitor
fi

# Starte eine neue tmux-Sitzung im Hintergrund
tmux new-session -d -s monitor

# Hole die Liste aller Screen-Sitzungen
sessions=$(screen -ls | awk '/[0-9]+\./{print $1}')

# Zähle die Anzahl der Screen-Sitzungen
session_count=$(echo "$sessions" | wc -l)

if [ "$session_count" -eq 0 ]; then
    echo "Keine aktiven Screen-Sitzungen gefunden."
    exit 1
fi

pane_index=0
for session in $sessions; do
    if [ $pane_index -gt 0 ]; then
        # Teile das Fenster horizontal oder vertikal (hier vertikal)
        tmux split-window -t monitor -v
        # Wechsle zum neuen Paneel
        tmux select-pane -t $pane_index
    fi
    # Führe den Befehl zum Anhängen an die Screen-Sitzung aus
    tmux send-keys -t monitor "screen -r $session" C-m
    pane_index=$((pane_index + 1))
done

# Ordne die Paneels an
tmux select-layout -t monitor tiled

# Verbinde dich mit der tmux-Sitzung
tmux attach-session -t monitor