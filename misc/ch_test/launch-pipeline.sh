#!/bin/bash
#
# Launch a pirate pipeline (sifter / grouper / server / fake X-engine /
# rpc_status) downstream-first, gating each launch on the previous process's
# readiness marker, then block so the children stay alive.
#
# Usage:  launch-pipeline.sh PLANFILE LOGDIR
#
# Run it from the pirate/ directory (config paths in the plan are relative),
# and run it in the BACKGROUND -- it does not return until the pipeline exits.
#
# The plan file supplies the command lines; this script supplies only the
# mechanics. That split is deliberate: /ch-test tells the agent to re-derive
# commands, ports and readiness markers from notes/quick_start.md and the
# source, because they change. Nothing in here hardcodes them.
#
# PLANFILE format -- one process per line, five '|'-separated fields:
#
#   name | readiness marker | count | timeout_sec | command
#
#   name     process name; also names the log (LOGDIR/name.log) and the
#            pidfile (LOGDIR/name.pid). check-logs.py expects the names
#            sifter, grouper, server, xengine, rpc_status.
#   marker   substring to wait for in THIS process's log before launching
#            the next one (fixed string, not a regex)
#   count    how many times the marker must appear (2 for a two-grouper
#            production run, where 'run toy_grouper' forks one child per
#            address; 1 otherwise)
#   timeout  seconds to wait for the marker before giving up (the
#            production server needs ~60 s to allocate its memory pools)
#   command  the command to run, split on whitespace (no quoting/globbing)
#
# Blank lines and lines starting with '#' are ignored. Example:
#
#   sifter  | waiting for grouper(s) to connect | 1 |  30 | pirate_frb run toy_sifter 127.0.0.1:7500
#   grouper | waiting for FrbServer to connect  | 1 |  30 | pirate_frb run toy_grouper -s 127.0.0.1:7500 127.0.0.1:7000
#   server  | server(s) started                 | 1 |  90 | pirate_frb run server configs/frb_server/toy.yml configs/dedispersion/toy.yml
#   xengine | FakeXEngine(s) running            | 1 |  30 | pirate_frb run fake_xengine -f -g 30 -s 127.0.0.1:7500 127.0.0.1:6000
#   rpc_status | Running get_status             | 1 |  30 | pirate_frb rpc status 127.0.0.1:6000
#
# While waiting for any marker, EVERY log in LOGDIR is watched for errors, so
# a startup failure is reported immediately instead of hanging the poll until
# the timeout.
#
# Writes LOGDIR/supervisor.log. Poll it for its terminal line:
#   "ALL READY"       every process launched and ready
#   "STARTUP FAILED"  a marker timed out, or an error appeared in some log
#
# On a startup failure, every process this script already launched is torn down
# (SIGINT, then SIGKILL for anything still alive after 30 s) before it exits, so
# a retry is not blocked by an orphan still holding a port or the hugepage pool.
#
# Exit status: 0 if the pipeline came up and later exited; 1 on startup
# failure; 2 on usage error.

set -u

if [ $# -ne 2 ]; then
    echo "usage: $0 PLANFILE LOGDIR" >&2
    exit 2
fi

PLAN=$1
LOG=$2

[ -r "$PLAN" ] || { echo "$0: cannot read plan file '$PLAN'" >&2; exit 2; }
mkdir -p "$LOG" || exit 2

SUP="$LOG/supervisor.log"
: > "$SUP"

# Python block-buffers redirected stdout, so readiness markers would sit
# invisible in the buffer without this.
export PYTHONUNBUFFERED=1

# Errors worth aborting a startup poll for. Deliberately broad: a false
# positive costs one confusing abort, a false negative costs a full timeout.
ERR_RE='Traceback|RuntimeError|Segmentation fault|Aborted|terminate called|Fatal Python error|CUDA error|bad_alloc|what\(\):'

# Bash starts async ('&') jobs in a non-job-control shell with SIGINT and
# SIGQUIT set to SIG_IGN, and CPython inherits and PRESERVES that SIG_IGN --
# so 'kill -INT' on a backgrounded pirate process is silently a no-op and the
# shutdown cascade never fires. This shim resets both signals to SIG_DFL and
# then exec()s the real command in the same pid, so the process behaves
# exactly as if Ctrl-C'd in its own terminal (which is how a human runs it).
SHIM_PY='import signal,os,sys; signal.signal(signal.SIGINT,signal.SIG_DFL); signal.signal(signal.SIGQUIT,signal.SIG_DFL); os.execvp(sys.argv[1],sys.argv[1:])'

sup() { echo "[$(date +%H:%M:%S)] $*" >> "$SUP"; }

trim() {
    local s=$1
    s=${s#"${s%%[![:space:]]*}"}
    s=${s%"${s##*[![:space:]]}"}
    printf '%s' "$s"
}

# Read the whole plan up front: a parse error should fail before anything is
# launched, and the launch loop must not hold the file open across 'wait'.
names=(); markers=(); counts=(); timeouts=(); cmds=()
lineno=0
while IFS= read -r line || [ -n "$line" ]; do
    lineno=$((lineno + 1))
    case "$(trim "$line")" in ''|'#'*) continue ;; esac

    IFS='|' read -r f_name f_marker f_count f_timeout f_cmd <<< "$line"
    if [ -z "${f_cmd:-}" ]; then
        echo "$0: $PLAN line $lineno: expected 5 '|'-separated fields" >&2
        exit 2
    fi
    names+=("$(trim "$f_name")")
    markers+=("$(trim "$f_marker")")
    counts+=("$(trim "$f_count")")
    timeouts+=("$(trim "$f_timeout")")
    cmds+=("$(trim "$f_cmd")")
done < "$PLAN"

if [ ${#names[@]} -eq 0 ]; then
    echo "$0: $PLAN contains no process lines" >&2
    exit 2
fi

# Processes this script has started, in launch order.
launched_pids=()
launched_names=()

# Tear down everything launched so far. Called when a startup fails partway
# through: the processes already up are still holding their ports -- and, for a
# production server, the entire hugepage pool and both GPUs -- so leaving them
# behind makes the NEXT attempt fail with an address-in-use (or an out-of-memory)
# that looks unrelated to the actual failure.
#
# SIGINT rather than SIGKILL, so each process runs its normal shutdown path; the
# shim above is what makes that work on a backgrounded job. A process that has
# exited but not been reaped stays visible in /proc as a zombie and still answers
# kill(pid, 0), so 'Z' in /proc/PID/stat counts as gone.
cleanup_launched() {
    [ ${#launched_pids[@]} -gt 0 ] || return 0
    sup "cleaning up ${#launched_pids[@]} launched process(es)"

    local i pid state alive tick

    for pid in "${launched_pids[@]}"; do
        kill -INT "$pid" 2>/dev/null
    done

    # Up to 30 s: a production server spends ~25 s releasing hugepages and GPU memory.
    for tick in $(seq 1 60); do
        alive=0
        for pid in "${launched_pids[@]}"; do
            state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)
            [ -n "$state" ] && [ "$state" != "Z" ] && alive=$((alive + 1))
        done
        [ "$alive" -eq 0 ] && break
        sleep 0.5
    done

    for i in "${!launched_pids[@]}"; do
        pid=${launched_pids[$i]}
        state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)
        if [ -n "$state" ] && [ "$state" != "Z" ]; then
            sup "  ${launched_names[$i]} (pid $pid) ignored SIGINT; sending SIGKILL"
            kill -KILL "$pid" 2>/dev/null
        fi
    done
}

# Poll $LOG/name.log until the marker has appeared $need times. Returns 1 on
# timeout, or as soon as any log shows an error.
waitmark() {
    local name=$1 marker=$2 need=$3 timeout=$4
    local logfile="$LOG/$name.log"
    local waited=0 ticks=$((timeout * 2)) n

    while [ $waited -lt $ticks ]; do
        n=$(grep -cF -- "$marker" "$logfile" 2>/dev/null || true)
        if [ "${n:-0}" -ge "$need" ]; then
            sup "OK   $name ready (${n}x '$marker')"
            return 0
        fi
        if grep -qE "$ERR_RE" "$LOG"/*.log 2>/dev/null; then
            sup "ERROR while waiting for $name:"
            grep -HnE "$ERR_RE" "$LOG"/*.log 2>/dev/null | tail -8 >> "$SUP"
            sup "STARTUP FAILED"
            return 1
        fi
        sleep 0.5
        waited=$((waited + 1))
    done

    sup "TIMEOUT after ${timeout}s waiting for $name marker '$marker' (need ${need}x)"
    sup "STARTUP FAILED"
    return 1
}

for i in "${!names[@]}"; do
    name=${names[$i]}
    read -ra argv <<< "${cmds[$i]}"

    python3 -c "$SHIM_PY" "${argv[@]}" > "$LOG/$name.log" 2>&1 &
    echo $! > "$LOG/$name.pid"
    launched_pids+=("$!")
    launched_names+=("$name")
    sup "launched $name pid=$! : ${cmds[$i]}"

    if ! waitmark "$name" "${markers[$i]}" "${counts[$i]}" "${timeouts[$i]}"; then
        cleanup_launched
        exit 1
    fi

    # Settle: the marker means "listening", but the process may still be
    # finishing setup its peer will immediately exercise.
    sleep 1
done

sup "ALL READY"
wait
sup "all children exited"
