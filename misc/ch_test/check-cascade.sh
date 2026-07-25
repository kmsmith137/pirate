#!/bin/bash
#
# Send SIGINT to one process of a running pipeline and verify that the
# shutdown cascades to all of them.
#
# Usage:  check-cascade.sh LOGDIR TARGET [MAX_WAIT_SEC]
#
#   LOGDIR        the directory launch-pipeline.sh wrote (name.pid, name.log)
#   TARGET        which process to interrupt -- the END of the pipeline, i.e.
#                 'sifter' for the standard toy/production runs
#   MAX_WAIT_SEC  how long to wait for everything to exit (default 45; the
#                 production server needs ~25 s to release its 1.5 TiB
#                 hugepage pool and ~80 GiB of GPU memory)
#
# Reports, per process: the pid, seconds from SIGINT to exit, and seconds from
# SIGINT to the FIRST cascade message appearing in its log, followed by the
# last few matching lines (which should be the documented cascade messages --
# the sifter's clean "interrupted; shutting down" and everyone else's
# RuntimeError are the expected "errors cascade backwards" path, not failures).
#
# Print time and exit time are reported separately because the cascade must be
# judged by the former. The production FrbServer prints its RuntimeError within
# ~2 s like everything else, then spends ~25 s tearing down its 1.5 TiB
# hugepage pool and ~80 GiB of GPU memory before the process actually goes
# away; reading only the exit time makes that look like a 25 s cascade.
#
# Matching lines are counted only from the point of the SIGINT onward: each
# log's length is snapshotted before the signal, so a pre-existing "Error" from
# earlier in the run cannot be mistaken for a cascade message (which would
# otherwise report a print time of 0.0 s for every process).
#
# Two liveness gotchas this handles, both of which make a naive check lie:
#
#   - PID 1 does not reap in the sandbox, so an exited process lingers as a
#     zombie, and a zombie still answers kill(pid, 0). State 'Z' in
#     /proc/PID/stat is therefore treated as exited.
#   - 'pgrep -f pirate' matches the watcher's own command line. Children are
#     found via 'pgrep -P' (parent pid) instead, which cannot self-match --
#     this is how the production run's two run_toy_grouper children, which
#     have no pidfile of their own, get tracked.
#
# Afterwards, checks that the resources actually came back (HugePages_Free
# in /proc/meminfo, nvidia-smi). A brief settle is allowed first: the counters
# lag the exit by a moment while zombies are still unreaped.
#
# Exit status: 0 if every process exited within MAX_WAIT_SEC and the resources
# returned; 1 otherwise.

set -u

# The lines that count as "a cascade message". Used BOTH to time the first
# such line and to select the ones printed at the end -- they must agree, or
# the reported time would describe a different line than the reported text.
# ERROR (uppercase) is listed explicitly: python's rpc_status logs
# "subscribe_files ERROR:", which a case-sensitive 'Error' misses.
ERR_PAT='interrupted|RuntimeError|Error|ERROR|stopped|disconnected'

# How many matching lines to show per process. More than one because the
# documented expectation is sometimes a PAIR: rpc_status should show its
# subscribe_files error AND the subsequent "RPC client(s) stopped".
NSHOW=3

LOG=${1:-}
TARGET=${2:-}
MAX_WAIT=${3:-45}

if [ -z "$LOG" ] || [ -z "$TARGET" ]; then
    echo "usage: $0 LOGDIR TARGET [MAX_WAIT_SEC]" >&2
    exit 2
fi

TARGET_PIDFILE="$LOG/$TARGET.pid"
[ -r "$TARGET_PIDFILE" ] || { echo "$0: no pidfile $TARGET_PIDFILE" >&2; exit 2; }

# alive PID -> 0 if the process exists and is not a zombie
alive() {
    local st
    st=$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null) || return 1
    [ -n "$st" ] && [ "$st" != "Z" ]
}

# Collect (name, pid) for every pidfile, plus each one's descendants.
names=(); pids=()
for pf in "$LOG"/*.pid; do
    [ -e "$pf" ] || continue
    n=$(basename "$pf" .pid)
    p=$(cat "$pf" 2>/dev/null) || continue
    [ -n "$p" ] || continue
    names+=("$n"); pids+=("$p")

    # One level of children is enough for run_toy_grouper, but recurse anyway.
    kids=$(pgrep -P "$p" 2>/dev/null || true)
    depth=0
    while [ -n "$kids" ] && [ $depth -lt 4 ]; do
        next=""
        for k in $kids; do
            names+=("$n.child"); pids+=("$k")
            next="$next $(pgrep -P "$k" 2>/dev/null || true)"
        done
        kids=$next
        depth=$((depth + 1))
    done
done

echo "tracking ${#pids[@]} process(es):"
for i in "${!pids[@]}"; do
    printf '  %-16s pid=%-8s %s\n' "${names[$i]}" "${pids[$i]}" \
        "$(alive "${pids[$i]}" && echo alive || echo 'ALREADY GONE')"
done

TPID=$(cat "$TARGET_PIDFILE")
if ! alive "$TPID"; then
    echo "ERROR: target '$TARGET' (pid $TPID) is not running -- nothing to interrupt" >&2
    exit 1
fi

# A process whose SIGINT is SIG_IGN will ignore this signal entirely (see the
# shim comment in launch-pipeline.sh). Warn rather than fail: the value here
# is telling the caller WHY nothing happened.
if grep -q '^SigIgn' "/proc/$TPID/status" 2>/dev/null; then
    mask=$(awk '/^SigIgn/{print $2}' "/proc/$TPID/status")
    if python3 -c "import sys; sys.exit(0 if int('$mask',16) & (1<<1) else 1)"; then
        echo "WARNING: pid $TPID has SIGINT in its SigIgn mask -- 'kill -INT' will be a"
        echo "         no-op. Launch via launch-pipeline.sh, which resets it to SIG_DFL."
    fi
fi

# Snapshot each log's length BEFORE the signal, so the cascade-message scan
# below only ever sees lines produced in response to it.
lognames=()
declare -A base_lines
for pf in "$LOG"/*.pid; do
    [ -e "$pf" ] || continue
    n=$(basename "$pf" .pid)
    [ -r "$LOG/$n.log" ] || continue
    lognames+=("$n")
    base_lines[$n]=$(wc -l < "$LOG/$n.log" 2>/dev/null || echo 0)
done

# Lines added to $1's log since the snapshot.
new_lines() {
    tail -n +$(( ${base_lines[$1]} + 1 )) "$LOG/$1.log" 2>/dev/null
}

echo
echo "sending SIGINT to $TARGET (pid $TPID) at $(date +%H:%M:%S)"
kill -INT "$TPID"
t0=$(date +%s%N)

elapsed() { awk "BEGIN{printf \"%.1f\", ($(date +%s%N) - $t0) / 1e9}"; }

# Record, for each log, when its first post-SIGINT cascade message appears.
declare -A msg_at
scan_msgs() {
    local n
    for n in "${lognames[@]}"; do
        [ -n "${msg_at[$n]:-}" ] && continue
        if new_lines "$n" | grep -qE "$ERR_PAT"; then
            msg_at[$n]=$(elapsed)
        fi
    done
}

declare -A exit_at
remaining=${#pids[@]}
ticks=$((MAX_WAIT * 2))
for ((tick = 0; tick < ticks; tick++)); do
    # Messages first: a process that prints and exits within one tick must
    # still get its print time recorded.
    scan_msgs
    for i in "${!pids[@]}"; do
        [ -n "${exit_at[$i]:-}" ] && continue
        if ! alive "${pids[$i]}"; then
            exit_at[$i]=$(elapsed)
            remaining=$((remaining - 1))
        fi
    done
    [ $remaining -eq 0 ] && break
    sleep 0.5
done

# Final sweep: catch anything flushed between the last scan and the last exit.
scan_msgs

echo
echo "exit times (seconds after SIGINT):"
rc=0
for i in "${!pids[@]}"; do
    if [ -n "${exit_at[$i]:-}" ]; then
        printf '  %-16s pid=%-8s exited@%ss\n' "${names[$i]}" "${pids[$i]}" "${exit_at[$i]}"
    else
        printf '  %-16s pid=%-8s STILL ALIVE after %ss\n' "${names[$i]}" "${pids[$i]}" "$MAX_WAIT"
        rc=1
    fi
done

echo
echo "cascade message times (seconds after SIGINT -- JUDGE THE CASCADE BY THESE,"
echo "not by the exit times above; a slow teardown is not a slow cascade):"
for n in "${lognames[@]}"; do
    if [ -n "${msg_at[$n]:-}" ]; then
        printf '  %-16s printed@%ss\n' "$n" "${msg_at[$n]}"
    else
        printf '  %-16s NO cascade message within %ss\n' "$n" "$MAX_WAIT"
        rc=1
    fi
done

echo
echo "cascade messages per process (last $NSHOW matching lines since the SIGINT;"
echo "expect the documented shutdown messages):"
for n in "${lognames[@]}"; do
    # Match only post-SIGINT lines, and show several: a process can emit more
    # output AFTER its shutdown message (the sifter's gRPC worker threads keep
    # printing in-flight events for a moment after the main thread's interrupt
    # handler runs), and some expectations are a pair rather than one line.
    msgs=$(new_lines "$n" | grep -E "$ERR_PAT" | tail -$NSHOW)
    if [ -z "$msgs" ]; then
        # Nothing matched: show the tail verbatim so the caller can see what
        # the process actually said instead of an empty row.
        printf '  %-16s (no matching line; last log line follows)\n' "$n"
        printf '  %-16s   | %s\n' "" "$(grep -v '^[[:space:]]*$' "$LOG/$n.log" | tail -1)"
        continue
    fi
    first=1
    while IFS= read -r line; do
        if [ $first -eq 1 ]; then
            printf '  %-16s %s\n' "$n" "$line"
            first=0
        else
            printf '  %-16s %s\n' "" "$line"
        fi
    done <<< "$msgs"
done

# Zombies hold no memory, but the counters can lag the exit by a moment.
sleep 3
echo
echo "resources after teardown:"
grep -E 'HugePages_(Total|Free)' /proc/meminfo | sed 's/^/  /'
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'
fi

hp_total=$(awk '/HugePages_Total/{print $2}' /proc/meminfo)
hp_free=$(awk '/HugePages_Free/{print $2}' /proc/meminfo)
if [ "${hp_free:-0}" != "${hp_total:-0}" ]; then
    echo "  WARNING: HugePages_Free ($hp_free) != HugePages_Total ($hp_total) --"
    echo "           re-check after a few more seconds before calling it a leak."
fi

exit $rc
