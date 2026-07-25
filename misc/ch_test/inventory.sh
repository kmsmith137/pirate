#!/bin/bash
#
# Emit the acquisition inventory table for the /ch-test final report: one row
# per directory, with size, file count and chunk range.
#
# Usage:  inventory.sh [--since EPOCH|FILE] DIR [DIR ...]
#
#   --since   mark rows created at or after this time as belonging to THIS
#             sweep, and everything older as pre-existing. Takes either an
#             epoch-seconds value or a file whose mtime is used (so the agent
#             can just 'touch $SCRATCH/sweep-start' at the beginning of the
#             run and pass it here). Without it, every row is reported
#             unclassified, and the agent has to work out by hand which
#             directories predate the sweep.
#
# Typical call at the end of a sweep:
#
#   inventory.sh --since $SCRATCH/sweep-start ~/pirate_toy /mnt/cs00/data/$USER
#
# The sweep writes tens of GB and nothing here deletes anything -- the user
# decides what to keep. The point of the "pre-existing" column is to let them
# clean up selectively.

set -u

SINCE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --since)
            [ $# -ge 2 ] || { echo "$0: --since needs an argument" >&2; exit 2; }
            if [ -e "$2" ]; then
                SINCE=$(stat -c %Y "$2")
            else
                SINCE=$2
            fi
            shift 2
            ;;
        -h|--help)
            sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) break ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "usage: $0 [--since EPOCH|FILE] DIR [DIR ...]" >&2
    exit 2
fi

echo "| Path | Size | Contents | Origin |"
echo "|------|------|----------|--------|"

for root in "$@"; do
    [ -d "$root" ] || { echo "| $root | - | (no such directory) | - |"; continue; }
    for d in "$root"/*/; do
        [ -d "$d" ] || continue
        d=${d%/}

        size=$(du -sh "$d" 2>/dev/null | cut -f1)
        nfiles=$(find "$d" -maxdepth 1 -type f | wc -l)

        # Acqdir frames are named frame_b{beam}_t{chunk}.asdf, so 't[0-9]+'
        # picks out the chunk index and nothing else.
        range=$(find "$d" -maxdepth 1 -type f -name '*.asdf' -printf '%f\n' 2>/dev/null \
                | grep -oE 't[0-9]+' | tr -d 't' | sort -n \
                | awk 'NR==1{f=$0} {l=$0} END{if (NR) printf "chunks %s-%s", f, l}')
        beams=$(find "$d" -maxdepth 1 -type f -name '*.asdf' -printf '%f\n' 2>/dev/null \
                | grep -oE '_b[0-9]+_' | tr -d '_b' | sort -nu | tr '\n' ',' | sed 's/,$//')

        contents="$nfiles files"
        [ -n "$range" ] && contents="$contents, $range"
        [ -n "$beams" ] && contents="$contents, beam(s) $beams"

        origin="-"
        if [ -n "$SINCE" ]; then
            mtime=$(stat -c %Y "$d" 2>/dev/null || echo 0)
            if [ "$mtime" -ge "$SINCE" ]; then origin="this sweep"; else origin="pre-existing"; fi
        fi

        echo "| $d | $size | $contents | $origin |"
    done
done

echo
echo "Note: rows marked 'this sweep' were created by this run. Annotate any that"
echo "were throwaway scratch (a debugging experiment rather than the step-4 /"
echo "step-5 acquisitions) so the user can clean up selectively. Delete nothing."
