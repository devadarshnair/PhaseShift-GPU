#!/usr/bin/env bash
# PhaseShift low-load wrapper. Same contract as the user's src/bench/heavy.sh, extended.
#
#   lowload.sh [--rss GB] [--mem GB] [--label L] [--log F] -- cmd [args...]
#
# Every heavy job (make, blackbox, Verilator, Yosys, SimX sweeps, Python jobs expected > 1 GB) goes through this.
#   1. Takes the machine-wide flock ${PS_LOCK:-/tmp/ml-hardware-bench.lock} -- the SAME lock as heavy.sh, so
#      PhaseShift never overlaps the user's other benches (PS_LOCK exists only for the unit test).
#   2. Admission: after taking the lock, reads MemAvailable from /proc/meminfo. The job is admitted iff
#        MemAvailable >= RSS + 0.75 GB
#      where RSS (expected peak resident set, GB) = --rss if given, else 1.2 x the largest maxrss_kb recorded in
#      the ledger for the same label prefix (the part of the label before the first ':'), else 1.5 GB.
#      The 0.75 GB margin leaves headroom for the user's own processes and the page cache (risk K1).
#      If not admitted, the lock is RELEASED (so the user's benches are not blocked), the wrapper sleeps
#      PS_ADMIT_SLEEP_S (default 60 s) and retries. After PS_ADMIT_WAIT_S (default 7200 s = 2 h) without
#      admission -- including time spent waiting for the lock -- it exits 75 (= DEFERRED, not a failure).
#   3. Appends a start line to the ledger, runs the job as a CHILD (no exec, so the end line can be written):
#        nice -n 19 ionice -c3 prlimit --as=<MEM GiB> -- /usr/bin/time -v -o <timefile> cmd
#      (with --log F: stdout+stderr are tee'd to F; pipefail propagates the job's non-zero status),
#      while a background loop touches $PS_RESULTS_DIR/$PS_STEP/.heartbeat every PS_HEARTBEAT_S (default 300 s)
#      if PS_STEP is set. Releases the lock, appends the end line, exits with the job's status.
#      The job runs in its own session and inherits the lock fd (lock held while any job process lives, even if the
#      wrapper is SIGKILLed); INT/TERM/HUP to the wrapper are forwarded to the job's process group, the wrapper waits
#      until the whole group is gone (TERM, then KILL after PS_KILL_GRACE_S, default 10 s), then the end line is
#      written and the wrapper exits 128+signal; the wrapper only closes its lock fd (never unlocks the shared lock); the heartbeat loop exits within ~1 s of the wrapper's death.
#      (Details and the daemon caveat at the run section below.)
#   4. --mem (virtual-address cap, integer GiB) defaults to PS_MEM_GB=4. NOTE: prlimit --as caps VIRTUAL ADDRESS
#      SPACE, not RSS (risk K2): LLVM/Verilator may reserve large VA and fail spuriously under a tight cap, so the
#      cap is deliberately separate from admission, which uses the EXPECTED RSS (--rss), not the VA cap.
#      MAKEFLAGS=-j1 and THREADS=1 (plus OMP/MKL/OPENBLAS_NUM_THREADS=1) are forced for the job; callers cannot
#      raise them through this wrapper.
#
# Ledger: $PS_RESULTS_DIR/lowload_ledger.tsv (PS_RESULTS_DIR defaults to $PS/results), tab-separated:
#   start <pid> <label> <rss_req_gb> <cmd> <iso-time>
#   end   <pid> <label> <exit> <maxrss_kb> <wall_s> <iso-time>
# <pid> is this wrapper's PID (pairs start/end lines). The /usr/bin/time -v file is <log>.time with --log, else
# $PS_RESULTS_DIR/logs/lowload_time/<label>.<pid>.time (git-ignored via /results/**/logs/).
# Exit status: the job's, 75 = DEFERRED (not admitted in time), 2 = usage error.
set -o pipefail

PS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${PS_RESULTS_DIR:-$PS_ROOT/results}"
LOCK="${PS_LOCK:-/tmp/ml-hardware-bench.lock}"
LEDGER="$RESULTS/lowload_ledger.tsv"
ADMIT_WAIT_S="${PS_ADMIT_WAIT_S:-7200}"
ADMIT_SLEEP_S="${PS_ADMIT_SLEEP_S:-60}"
HEARTBEAT_S="${PS_HEARTBEAT_S:-300}"
KILL_GRACE_S="${PS_KILL_GRACE_S:-10}"   # TERM -> KILL grace for the job group on a forwarded signal
MEM_GB="${PS_MEM_GB:-4}"
MARGIN_GB=0.75
DEFAULT_RSS_GB=1.5
RSS_GB=""
LABEL="job"
LOG=""

usage() { echo "usage: lowload.sh [--rss GB] [--mem GB] [--label L] [--log F] -- cmd [args...]" >&2; exit 2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --rss|--mem|--label|--log)
            # a value flag given last (no value) is a usage error, not an endless loop
            [ $# -ge 2 ] || { echo "lowload: $1 needs a value" >&2; usage; }
            case "$1" in
                --rss) RSS_GB="$2" ;; --mem) MEM_GB="$2" ;; --label) LABEL="$2" ;; --log) LOG="$2" ;;
            esac
            shift 2 ;;
        --) shift; break ;;
        *) usage ;;
    esac
done
[ $# -gt 0 ] || usage
[[ "$HEARTBEAT_S" =~ ^[1-9][0-9]*$ ]] || { echo "lowload: PS_HEARTBEAT_S must be a positive integer" >&2; exit 2; }
[[ "$KILL_GRACE_S" =~ ^[0-9]+$ ]] || { echo "lowload: PS_KILL_GRACE_S must be a non-negative integer" >&2; exit 2; }
[[ "$MEM_GB" =~ ^[0-9]+$ ]] || { echo "lowload: --mem must be an integer number of GiB" >&2; exit 2; }
[ -z "$RSS_GB" ] || [[ "$RSS_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "lowload: --rss must be a number (GB)" >&2; exit 2; }
LABEL="${LABEL//[$'\t\n']/_}"

mkdir -p "$RESULTS"

# Expected RSS (GB) when --rss is not given: 1.2 x largest recorded maxrss for the same label prefix, else 1.5 GB.
if [ -z "$RSS_GB" ]; then
    prefix="${LABEL%%:*}"
    RSS_GB=$(awk -F'\t' -v p="$prefix" -v d="$DEFAULT_RSS_GB" '
        # exact prefix match: the label part before the first ":" must EQUAL p (so S6 does not match S60:...)
        $1 == "end" && ((i = index($3, ":")) ? substr($3, 1, i - 1) : $3) == p && $5 ~ /^[0-9]+$/ { if ($5 > m) m = $5 }
        END { if (m > 0) printf "%.3f", 1.2 * m / 1048576; else print d }' "$LEDGER" 2>/dev/null || echo "$DEFAULT_RSS_GB")
fi

mem_available_gb() { awk '/^MemAvailable:/ { printf "%.3f", $2 / 1048576 }' /proc/meminfo; }

# --- memwatch autostart (optional machine-wide memory watchdog, if installed) ------------------------------------------
# Start the OOM guard once if it is not running ("running" = its singleton flock is held). Done before fd 9 is opened
# so the guard never inherits the machine lock. PS_NO_MEMWATCH=1 disables it (unit tests).
if [ -z "${PS_NO_MEMWATCH:-}" ] && flock -n /tmp/research-memwatch.lock true 2>/dev/null && [ -x ~/research/shared/memwatch.sh ]; then
    setsid nohup ~/research/shared/memwatch.sh >/dev/null 2>&1 </dev/null &
    echo "$(date -Is) autostart memwatch.sh (was not running; lowload.sh)" >> ~/research/shared/heavygate.log 2>/dev/null
fi

# --- re-entrancy --------------------------------------------------------------------------------------------------------
# held=1 when this lowload.sh already runs inside a lock that an outer holder owns:
#   (a) PS_LOWLOAD_HELD is set (an outer lowload.sh exported it to its job), or
#   (b) an ANCESTOR process holds an exclusive flock on $LOCK (e.g. qflock.sh / `flock LOCK cmd` outer hold; found
#       through /proc/locks: a FLOCK ... WRITE entry on the lock file's inode whose pid is one of our ancestors).
# In both cases waiting for the lock or for lane admission would deadlock on ourselves (exit 75 seen 2026-11-14), so
# FIFO + flock + admission are skipped; nice/ionice/prlimit/ledger/log still apply.
held=0
ancestor_holds_lock() {
    local ino pid pids=" " lpid
    [ -e "$LOCK" ] || return 1
    ino=$(stat -c %i "$LOCK" 2>/dev/null) || return 1
    pid=$PPID
    while [ -n "$pid" ] && [ "$pid" -gt 1 ] 2>/dev/null; do
        pids="$pids$pid "
        pid=$(awk '/^PPid:/ { print $2 }' "/proc/$pid/status" 2>/dev/null)
    done
    while read -r lpid; do
        case "$pids" in *" $lpid "*) return 0 ;; esac
    done < <(awk -v ino="$ino" '$2 == "FLOCK" && $4 == "WRITE" { n = split($6, a, ":"); if (a[n] == ino) print $5 }' /proc/locks)
    return 1
}
if [ -n "${PS_LOWLOAD_HELD:-}" ]; then held=1
elif ancestor_holds_lock; then held=1
fi

# --- admission loop -------------------------------------------------------------------------------------------
deadline=$(( $(date +%s) + ADMIT_WAIT_S ))
admitted=0
if [ "$held" -eq 1 ]; then
    admitted=1
    echo "lowload: lock already held by an outer holder (label=$LABEL); skipping queue/admission" >&2
else
    exec 9>>"$LOCK"
fi
# Machine CPU lane (optional, if the shared lane scripts are installed): FIFO queue + MEMORY admission via ~/research/shared/laneadmit.sh --
# several jobs may share the lane (cap 3) while the shared RAM ledger admits them; this job holds the lane lock
# SHARED and is registered in /tmp/ml-hardware-mem.d/<pid> until it exits. The per-step inner PS_LOCK keeps the
# old exclusive loop below.
if [ "$admitted" -eq 0 ] && [ "$LOCK" = /tmp/ml-hardware-bench.lock ] && [ -r ~/research/shared/laneadmit.sh ]; then
    . ~/research/shared/memledger.sh; . ~/research/shared/fifoq.sh; . ~/research/shared/laneadmit.sh
    fifo_enter "$LOCK" "lowload:$LABEL"
    if lane_admit 9 cpu "$RSS_GB" "lowload:$LABEL" "$deadline"; then admitted=1
    else echo "lowload: DEFERRED -- not admitted within ${ADMIT_WAIT_S}s (label=$LABEL, ${LA_WHY:-queued})" >&2; exit 75; fi
fi
[ "$admitted" -eq 1 ] || while :; do
    remaining=$(( deadline - $(date +%s) ))
    [ "$remaining" -gt 0 ] || break
    if ! flock -n 9; then
        echo "lowload: waiting for lock $LOCK" >&2
        flock -w "$remaining" 9 || break
    fi
    avail=$(mem_available_gb)
    if [ "$LOCK" = /tmp/ml-hardware-bench.lock ] && [ -r ~/research/shared/memledger.sh ]; then
        # Shared RAM ledger (2026-11-14): admit only if MemAvailable minus other jobs' unclaimed reservations
        # >= RSS + 1.0 GB; writes /tmp/ml-hardware-mem.d/<pid> = "<GB> cpu <label>", swept when this pid exits.
        . ~/research/shared/memledger.sh
        if ml_try_admit "$$" "$RSS_GB" cpu "lowload:$LABEL"; then admitted=1; break; fi
        avail="$avail ($ML_LAST)"
    elif awk -v a="$avail" -v r="$RSS_GB" -v m="$MARGIN_GB" 'BEGIN { exit !(a >= r + m) }'; then
        admitted=1
        break
    fi
    flock -u 9   # release while waiting for memory, so other benches can run
    remaining=$(( deadline - $(date +%s) ))
    [ "$remaining" -gt 0 ] || break
    nap=$ADMIT_SLEEP_S; [ "$nap" -le "$remaining" ] || nap=$remaining
    echo "lowload: not admitted (MemAvailable ${avail} GB < ${RSS_GB} + ${MARGIN_GB} GB); retry in ${nap}s" >&2
    sleep "$nap"
done
if [ "$admitted" -ne 1 ]; then
    echo "lowload: DEFERRED -- not admitted within ${ADMIT_WAIT_S}s (label=$LABEL, rss=${RSS_GB} GB)" >&2
    exit 75
fi

# --- run --------------------------------------------------------------------------------------------------------
export PS_LOWLOAD_HELD="${PS_LOWLOAD_HELD:-$$}"   # D3: nested lowload.sh calls inside this job are re-entrant
export MAKEFLAGS=-j1 THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

if [ -n "$LOG" ]; then
    mkdir -p "$(dirname "$LOG")"
    export LL_LOG="$LOG"
    TIMEF="$LOG.time"
else
    mkdir -p "$RESULTS/logs/lowload_time"
    TIMEF="$RESULTS/logs/lowload_time/${LABEL//\//_}.$$.time"
fi

cmdstr="$*"; cmdstr="${cmdstr//[$'\t\n']/ }"
printf 'start\t%s\t%s\t%s\t%s\t%s\n' "$$" "$LABEL" "$RSS_GB" "$cmdstr" "$(date -Iseconds)" >> "$LEDGER"

# Lock ownership (decision, S2): the job INHERITS the lock fd 9, so the machine-wide lock stays held for as
# long as any process of the job is alive -- even if this wrapper is SIGKILLed (no trap can run then). Catchable
# signals (INT/TERM/HUP) are forwarded to the job's whole process group (the job runs in its own session via setsid),
# the wrapper waits for it, writes the end line and exits 128+signal, so a TERMed wrapper never orphans its job.
# Caveat: a job that leaves a daemon behind would keep the lock until the daemon exits; none of the job types run
# through this wrapper (make, Verilator, Yosys, SimX, Python) daemonizes. The heartbeat loop closes fd 9 and exits
# within ~1 s of the wrapper's death (it polls `kill -0 <wrapper pid>`), so a dead wrapper stops the heartbeat and
# step_lock.sh frees the step after its stale window.
WRAPPER_PID=$$
hb_pid=""
job_pid=""
status=""
ended=0
t0=$(date +%s.%N)

write_end() {   # idempotent: kill the heartbeat, drop OUR reference to the lock, append the end line once
    [ "$ended" -eq 1 ] && return
    ended=1
    if [ -n "$hb_pid" ]; then kill "$hb_pid" 2>/dev/null; wait "$hb_pid" 2>/dev/null; fi
    # Close fd 9, never `flock -u 9`: the lock belongs to the open file description that the job shares, so an
    # explicit unlock would free it for job processes that are still alive. Closing only drops this process's
    # reference; the lock is released when the last holder (wrapper or any surviving job process) closes it.
    exec 9>&-
    local t1 maxrss wall
    t1=$(date +%s.%N)
    maxrss=$(awk -F': ' '/Maximum resident set size/ { print $2 }' "$TIMEF" 2>/dev/null)
    [ -n "$maxrss" ] || maxrss=NA
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')
    printf 'end\t%s\t%s\t%s\t%s\t%s\t%s\n' "$WRAPPER_PID" "$LABEL" "${status:-NA}" "$maxrss" "$wall" "$(date -Iseconds)" >> "$LEDGER"
}

on_signal() {   # $1 = signal name, $2 = signal number
    trap '' INT TERM HUP
    # Liveness is checked on the whole process group (kill -0 -- -PGID), not on the leader: the leader is
    # /usr/bin/time (or the --log `bash -c ... | tee` shell), which dies on TERM at once while the real job may
    # ignore TERM or take seconds to clean up.
    if [ -n "$job_pid" ] && kill -0 -- -"$job_pid" 2>/dev/null; then
        echo "lowload: caught SIG$1, forwarding to job process group $job_pid" >&2
        # INT is sent as TERM: background commands of a non-interactive shell start with SIGINT ignored
        local fwd="$1"; [ "$fwd" = INT ] && fwd=TERM
        kill -"$fwd" -- -"$job_pid" 2>/dev/null
        local i
        for (( i = 0; i < KILL_GRACE_S * 10; i++ )); do kill -0 -- -"$job_pid" 2>/dev/null || break; sleep 0.1; done
        if kill -0 -- -"$job_pid" 2>/dev/null; then   # group still alive after the grace period
            echo "lowload: job group $job_pid still alive after ${KILL_GRACE_S}s, sending KILL" >&2
            kill -KILL -- -"$job_pid" 2>/dev/null
            for (( i = 0; i < 50; i++ )); do kill -0 -- -"$job_pid" 2>/dev/null || break; sleep 0.1; done
        fi
    fi
    if [ -n "$job_pid" ] && [ -z "$status" ]; then wait "$job_pid" 2>/dev/null; status=$?; fi
    [ -n "$status" ] || status=$(( 128 + $2 ))
    write_end
    exit $(( 128 + $2 ))
}
trap 'on_signal INT 2' INT
trap 'on_signal TERM 15' TERM
trap 'on_signal HUP 1' HUP
trap 'write_end' EXIT

if [ -n "${PS_STEP:-}" ]; then
    mkdir -p "$RESULTS/$PS_STEP"
    # 1-s poll so the loop notices the wrapper's death quickly; touches every HEARTBEAT_S seconds.
    ( n=0
      while kill -0 "$WRAPPER_PID" 2>/dev/null; do
          [ $(( n % HEARTBEAT_S )) -eq 0 ] && touch "$RESULTS/$PS_STEP/.heartbeat"
          n=$(( n + 1 ))
          sleep 1
      done ) 9>&- </dev/null >/dev/null 2>&1 &
    hb_pid=$!
fi

# The job: own session/process group (setsid; a background child of a non-interactive shell is not a group leader,
# so setsid does not fork and $! is the group id), stdin kept (explicit <&0), lock fd 9 inherited (see above).
# Signals that arrive while the job is being forked are only recorded (deferred traps) and handled right after
# job_pid is known, so there is no window in which a signal leaves the job unsignalled.
pending=""
trap 'pending="INT 2"' INT
trap 'pending="TERM 15"' TERM
trap 'pending="HUP 1"' HUP
if [ -n "$LOG" ]; then
    setsid bash -c 'set -o pipefail; "$@" 2>&1 | tee -a "$LL_LOG"' lowload-job \
        nice -n 19 ionice -c3 prlimit --as=$(( MEM_GB << 30 )) -- /usr/bin/time -v -o "$TIMEF" "$@" <&0 &
else
    setsid nice -n 19 ionice -c3 prlimit --as=$(( MEM_GB << 30 )) -- /usr/bin/time -v -o "$TIMEF" "$@" <&0 &
fi
job_pid=$!
trap 'on_signal INT 2' INT
trap 'on_signal TERM 15' TERM
trap 'on_signal HUP 1' HUP
[ -n "$pending" ] && on_signal $pending
wait "$job_pid"
status=$?
write_end
trap - EXIT
exit "$status"
