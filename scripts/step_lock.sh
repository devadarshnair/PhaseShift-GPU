#!/usr/bin/env bash
# PhaseShift step lock (S2, §5 resume protocol).
#
#   step_lock.sh acquire|release|status|touch <ID>
#
# Files (PS_RESULTS_DIR defaults to $PS/results; all git-ignored):
#   results/<ID>/.running.lock  flock target; held by a small detached holder process while the step is live
#   results/<ID>/.heartbeat     touched by the working session (`touch`, >= every 10 min), lowload.sh, vxrun.py
#   results/<ID>/.release       created by `release`; tells the holder to exit
#
# acquire  fails (exit 1) if the lock is held (flock -n probe). Otherwise it removes any stale .release, touches
#          .heartbeat, and only then starts the detached holder (setsid flock -n .running.lock <loop>), so a re-run
#          after an interrupted attempt (old heartbeat > 30 min stale) does not see its holder exit immediately.
#          It waits until the holder has written this call's token into the lock file (success, exit 0) or the
#          holder has died (lost a race to another acquirer, exit 1).
# holder   loop exits when .heartbeat is older than PS_STALE_S (default 1800 s = 30 min) or .release exists;
#          it checks .release every 1 s and the heartbeat age every clamp(PS_STALE_S/4, 1, 30) s. A dead session
#          therefore frees the step within ~30.5 min.
# release  creates .release and waits (up to 10 s) until the lock is free; exit 0 when free.
# status   prints `live` if the lock is held (flock -n fails), else `free`. Exit 0 either way.
# touch    updates .heartbeat (creates the step directory if needed).
set -uo pipefail

PS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${PS_RESULTS_DIR:-$PS_ROOT/results}"
STALE_S="${PS_STALE_S:-1800}"
POLL_S=$(( STALE_S / 4 )); [ "$POLL_S" -ge 1 ] || POLL_S=1; [ "$POLL_S" -le 30 ] || POLL_S=30

usage() { echo "usage: step_lock.sh acquire|release|status|touch <ID>" >&2; exit 2; }
[ $# -eq 2 ] || usage
cmd="$1"; id="$2"
[[ "$id" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "step_lock: bad step ID '$id'" >&2; exit 2; }
dir="$RESULTS/$id"; lock="$dir/.running.lock"; hb="$dir/.heartbeat"; rel="$dir/.release"

is_live() { [ -e "$lock" ] && ! flock -n "$lock" true 2>/dev/null; }

case "$cmd" in
    status)
        if is_live; then echo live; else echo free; fi
        ;;
    touch)
        mkdir -p "$dir" && touch "$hb"
        ;;
    acquire)
        mkdir -p "$dir"
        if is_live; then echo "step_lock: $id is live (lock held); not acquiring" >&2; exit 1; fi
        rm -f "$rel"
        touch "$hb"
        token="$$.$(date +%s%N)"
        # Detached holder: new session, no inherited stdio, takes the lock non-blocking (exits if someone else won).
        # Inside the loop: write the token into the lock file (content does not affect the flock), then poll.
        setsid flock -n "$lock" bash -c '
            lock=$1 hb=$2 rel=$3 stale=$4 poll=$5 token=$6
            printf "%s holder_pid=%s\n" "$token" "$$" > "$lock"
            n=0
            while :; do
                [ -e "$rel" ] && break
                if [ $(( n % poll )) -eq 0 ]; then
                    m=$(stat -c %Y "$hb" 2>/dev/null || echo 0)
                    [ $(( EPOCHSECONDS - m )) -gt "$stale" ] && break
                fi
                n=$(( n + 1 ))
                sleep 1
            done' holder "$lock" "$hb" "$rel" "$STALE_S" "$POLL_S" "$token" </dev/null >/dev/null 2>&1 9>&- &
        hpid=$!
        for i in $(seq 1 100); do   # up to ~10 s
            if grep -q "^$token " "$lock" 2>/dev/null && is_live; then echo "step_lock: acquired $id"; exit 0; fi
            # holder gone (lost the race): give up early, after a 2 s grace in case setsid forked
            if [ "$i" -gt 20 ] && ! kill -0 "$hpid" 2>/dev/null; then break; fi
            sleep 0.1
        done
        echo "step_lock: failed to acquire $id (another holder won or holder did not start)" >&2
        exit 1
        ;;
    release)
        [ -d "$dir" ] || { echo free; exit 0; }
        touch "$rel"
        end=$(( $(date +%s) + 10 ))
        while is_live; do
            [ "$(date +%s)" -lt "$end" ] || { echo "step_lock: $id still live after release request" >&2; exit 1; }
            sleep 0.2
        done
        rm -f "$rel"
        echo "step_lock: released $id"
        ;;
    *) usage ;;
esac
