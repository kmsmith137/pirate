#!/usr/bin/env python3
"""Cross-check offline-dedisperser output against the fake X-engine's truth.

Every FRB the fake X-engine injected into the streamed beam, inside the
acquired chunk range, should produce an SNR spike in the offline dedisperser's
output. This does the bookkeeping the /ch-test command describes in prose:
parse both logs, convert each truth event's fpga_timestamp to a time-chunk
index, and look for a spike inside a per-event window.

    check-truth.py --xengine-log X.log --dedisp-log D.log --beam 100 \
                   --freq-lo 300 --freq-hi 1500

The window runs from --window-before chunks before the truth tci out to the
end of the pulse's dispersion sweep,

    sweep_seconds = k_dm * DM * (f_lo^-2 - f_hi^-2) / 1000

(k_dm = 4.148808e6 ms MHz^2, pirate::constants::k_dm). A high-DM pulse sweeps
many chunks, so its spike can land far after the truth tci; --window-before
covers the other direction, where early-trigger trees fire before it. Adjacent
chunks also echo, because the offline dedisperser's peak-finding is a
placeholder that just reports each chunk's max.

Truth events in the first --warmup chunks of the acquisition are allowed to
have no spike: dedispersion sums there extend back past the start of the
acquired data (the documented "boundary effects near the beginning of the
acquisition"). Those are reported as 'warm', separately from hard misses.

Exit status 0 if every non-warmup truth event matched, 1 otherwise, and 2 if
the inputs could not be parsed. A run that parses no truth events, or no
dedisperser chunks, is an ERROR, never a vacuous pass -- if a log format
changes, this must fail loudly rather than report 0/0.
"""

import argparse
import re
import statistics
import sys

K_DM_MS = 4.148808e6   # pirate::constants::k_dm, in ms MHz^2

# "  beam 100  chunk   0 (tci= 45):  snr_max=  5.434"
RE_CHUNK = re.compile(
    r'beam\s+(\d+)\s+chunk\s+(\d+)\s+\(tci=\s*(\d+)\):\s+snr_max=\s*([\d.eE+-]+)')
# "  nfreq=28160, nt_in=2048, ntrees=4, dtype=float16, time_sample_ms=0.9984"
RE_DEDISP_HDR = re.compile(r'nt_in=(\d+).*?time_sample_ms=([\d.eE+-]+)')
# "    injected FRB: beam_id=100, dm=1310, fpga_timestamp=52428800, ..."
RE_INJECT = re.compile(
    r'injected FRB: beam_id=(\d+), dm=([\d.eE+-]+),\s*fpga_timestamp=(\d+)')
# "[10.222.3.5:6000]   time_samples_per_chunk = 2048"
RE_TSPC = re.compile(r'time_samples_per_chunk\s*=\s*(\d+)')
RE_SPFTS = re.compile(r'seq_per_frb_time_sample\s*[:=]\s*(\d+)')


def die(msg):
    print(f"check-truth: ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def parse_dedisp(path, beam):
    """-> (sorted tci list, {tci: snr_max}, samples_per_chunk, time_sample_ms)"""
    per, nt_in, dt_ms = {}, None, None
    beams_seen = set()
    with open(path) as f:
        for line in f:
            m = RE_DEDISP_HDR.search(line)
            if m and nt_in is None:
                nt_in, dt_ms = int(m.group(1)), float(m.group(2))
            m = RE_CHUNK.search(line)
            if m:
                b = int(m.group(1))
                beams_seen.add(b)
                if b == beam:
                    per[int(m.group(3))] = float(m.group(4))
    if not per:
        die(f"no 'snr_max' chunk lines for beam {beam} in {path}"
            + (f" (beams present: {sorted(beams_seen)})" if beams_seen else
               " -- no chunk lines at all; has the output format changed?"))
    if nt_in is None:
        die(f"could not find the 'nt_in=..., time_sample_ms=...' header in {path}")
    return sorted(per), per, nt_in, dt_ms


def parse_xengine(path, beam):
    """-> (list of (tci_numerator=fpga, dm), seqs_per_chunk)"""
    tspc = spfts = None
    events, total = [], 0
    with open(path) as f:
        for line in f:
            if tspc is None:
                m = RE_TSPC.search(line)
                if m:
                    tspc = int(m.group(1))
            if spfts is None:
                m = RE_SPFTS.search(line)
                if m:
                    spfts = int(m.group(1))
            m = RE_INJECT.search(line)
            if m:
                total += 1
                if int(m.group(1)) == beam:
                    events.append((int(m.group(3)), float(m.group(2))))
    if total == 0:
        die(f"no 'injected FRB:' lines in {path} -- was the fake X-engine run "
            f"with -f (inject FRBs)? If it was, the log format has changed.")
    if not events:
        die(f"{total} injected FRBs in {path}, but none on beam {beam}")
    if tspc is None or spfts is None:
        die(f"could not derive seqs_per_chunk from {path}: need both "
            f"'time_samples_per_chunk' and 'seq_per_frb_time_sample' "
            f"(printed at fake X-engine startup)")
    return events, tspc * spfts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xengine-log', required=True)
    ap.add_argument('--dedisp-log', required=True)
    ap.add_argument('--beam', type=int, required=True,
                    help='the streamed beam id (the one in the acqdir)')
    ap.add_argument('--freq-lo', type=float, required=True,
                    help='band bottom in MHz (zone_freq_edges[0] of the '
                         'dedispersion config: 400 toy, 300 production)')
    ap.add_argument('--freq-hi', type=float, required=True,
                    help='band top in MHz (zone_freq_edges[-1]: 800 toy, '
                         '1500 production)')
    ap.add_argument('--warmup', type=int, default=8, metavar='N',
                    help='truth events in the first N chunks of the acquisition '
                         'may legitimately have no spike (default 8)')
    ap.add_argument('--window-before', type=int, default=4, metavar='N',
                    help='also accept a spike up to N chunks BEFORE the truth '
                         'tci (early-trigger trees, adjacent-chunk echoes; '
                         'default 4)')
    ap.add_argument('--nsigma', type=float, default=4.0,
                    help='spike threshold = max(--floor, median + nsigma*stdev) '
                         'over all chunks (default 4)')
    ap.add_argument('--floor', type=float, default=10.0,
                    help='lower bound on the spike threshold (default 10)')
    args = ap.parse_args()

    if args.freq_lo <= 0 or args.freq_hi <= args.freq_lo:
        die(f"need 0 < freq_lo < freq_hi, got {args.freq_lo}, {args.freq_hi}")

    tcis, per, nt_in, dt_ms = parse_dedisp(args.dedisp_log, args.beam)
    truth_all, seqs_per_chunk = parse_xengine(args.xengine_log, args.beam)

    lo, hi = tcis[0], tcis[-1]
    vals = [per[t] for t in tcis]
    med = statistics.median(vals)
    # Population stdev is inflated by the spikes themselves, which RAISES the
    # threshold -- the conservative direction for a detection test.
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    thresh = max(args.floor, med + args.nsigma * sd)

    print(f"dedisperser: beam {args.beam}, {len(tcis)} chunks, tci [{lo}, {hi}]")
    print(f"  snr_max median={med:.3f} stdev={sd:.3f} max={max(vals):.3f}")
    print(f"  spike threshold = max({args.floor}, {med:.3f} + {args.nsigma}*{sd:.3f})"
          f" = {thresh:.3f}  ({sum(1 for v in vals if v >= thresh)} chunks above)")
    print(f"x-engine:    seqs_per_chunk = {seqs_per_chunk}, "
          f"{len(truth_all)} injection(s) on beam {args.beam}")

    truth = sorted((fpga // seqs_per_chunk, dm) for fpga, dm in truth_all)
    in_range = [(t, dm) for (t, dm) in truth if lo <= t <= hi]
    if not in_range:
        die(f"none of the {len(truth)} beam-{args.beam} injections "
            f"(tci {truth[0][0]}..{truth[-1][0]}) fall inside the acquired "
            f"range [{lo}, {hi}] -- wrong acqdir, wrong beam, or wrong log pair?")
    print(f"  {len(in_range)} injection(s) inside the acquired range\n")

    # chunks spanned by the pulse's dispersion sweep
    samples_per_ms = 1.0 / dt_ms
    def sweep_chunks(dm):
        delay_ms = K_DM_MS * dm * (args.freq_lo ** -2 - args.freq_hi ** -2)
        return int(delay_ms * samples_per_ms / nt_in) + 1

    matched = warm = hard = 0
    misses = []
    for tci, dm in in_range:
        swc = sweep_chunks(dm)
        a, b = tci - args.window_before, tci + swc + args.window_before
        win = [t for t in tcis if a <= t <= b]
        peak = max((per[t] for t in win), default=0.0)
        at = max(win, key=lambda t: per[t]) if win else None

        if peak >= thresh:
            matched += 1
            tag = 'OK  '
        elif (tci - lo) < args.warmup:
            warm += 1
            tag = 'warm'
        else:
            hard += 1
            tag = 'MISS'
            misses.append((tci, dm, swc, round(peak, 2)))
        print(f"  {tag} tci={tci:6d} dm={dm:9.2f} sweep~{swc:4d}ch "
              f"window[{a},{b}] peak={peak:7.2f}"
              + (f"@tci={at}" if at is not None else ""))

    total = len(in_range)
    print(f"\nSUMMARY: {matched}/{total} matched, {warm} warmup-excused, "
          f"{hard} hard miss(es)")
    if misses:
        print("HARD MISSES (tci, dm, sweep_chunks, best_peak):")
        for m in misses:
            print(f"  {m}")
    print("RESULT:", "PASS" if hard == 0 else "FAIL")
    return 0 if hard == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
