#!/bin/bash
# Keep the Condensate laptop observer running. Called by cron every minute.
# If it's already running, does nothing.
PIDFILE=/tmp/condensate_observer_laptop.pid
LOGFILE=/home/josh/Condensate/laptop_observer.log

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        exit 0  # already running
    fi
fi

# Start fresh
nohup python3 /home/josh/Condensate/condensate_observer_laptop.py \
    >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
