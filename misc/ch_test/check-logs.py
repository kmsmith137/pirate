#!/usr/bin/env python3
"""Scan a pipeline log directory for the "looks reasonable" checks.

Mechanizes the per-log checks the /ch-test command lists: the ones that are a
matter of parsing and arithmetic, not judgment. Judgment calls (is this
failure a bug in the code or in the test procedure? is this SNR plausible?)
stay with the agent -- this prints the numbers it needs to make them.

    check-logs.py --logdir LOGDIR [--cascade] [--acqdir NAME]

Expects the log names launch-pipeline.sh writes: server.log, grouper.log,
sifter.log, xengine.log, rpc_status.log. Missing logs are reported and
skipped, not treated as failures (a partial pipeline is a legitimate run).

Checks, per log:

  server      per-chunk lines exist and ichunk advances monotonically within
              each beamset; the FIRST line is well-formed (a truncated first
              line has been a real bug)
  grouper     coarse_snr_max baseline range over nevents=0 chunks, and the
              spike count -- baseline should sit around 5-9, spikes near the
              injected SNR
  xengine     injected-FRB count, the distinct SNR values, and per-beam counts
  sifter      BOTH event streams present (FROM_SIMULATOR truth and grouper
              search events), and what fraction of search detections have a DM
              within 2x of the nearest same-beam truth injection
  rpc_status  every ring-buffer counter monotonically non-decreasing, per
              server address; connection count; number of streamed filenames

With --cascade (the run ended with a deliberate SIGINT), errors are EXPECTED
in the tail of each log -- that is the documented "errors cascade backwards"
path. Errors are then a failure only if they appear before the cascade
region. Without --cascade, any error is a failure.

Exit status 0 if every check passed, 1 if any hard check failed, 2 on usage
error. A log that exists but yields zero parseable lines is a FAILURE, not a
pass: a format change must be loud.
"""

import argparse
import os
import re
import statistics
import sys

ERR_RE = re.compile(
    r'Traceback|RuntimeError|Segmentation fault|Aborted|terminate called|'
    r'Fatal Python error|CUDA error|bad_alloc|AssertionError|ERROR')
# Lines that merely contain the word "error" in a benign field.
ERR_SKIP_RE = re.compile(r'errored = 0|errored=0|error-check|no error')

# "FrbServer: beamset=0, ichunk=10, fpga=[3993600:4392960], Gbps=11.54, ..."
RE_SERVER = re.compile(r'FrbServer: beamset=(\d+), ichunk=(\d+), fpga=\[(\d+):(\d+)\]')
# "toy_grouper: beamset=0, ichunk=9, fpga=[..], coarse_snr_max=6.805, nevents=0"
RE_GROUPER = re.compile(r'coarse_snr_max=([\d.eE+-]+), nevents=(\d+)')
# "    injected FRB: beam_id=100, dm=1310, fpga_timestamp=..., snr=30"
RE_INJECT = re.compile(
    r'injected FRB: beam_id=(\d+), dm=([\d.eE+-]+),\s*fpga_timestamp=(\d+).*?snr=([\d.eE+-]+)')
# "toy_sifter FROM_SIMULATOR beamset=0 fpga=[..] coarse_snr_max=0"
RE_SIFTER_HDR = re.compile(r'^toy_sifter (FROM_SIMULATOR )?beamset=')
# "    beam_id=44, dm=2211, fpga_timestamp=472355961, ..., snr=30"
RE_SIFTER_EV = re.compile(
    r'^\s+beam_id=(\d+), dm=([\d.eE+-]+), fpga_timestamp=(\d+).*?snr=([\d.eE+-]+)')
# "[127.0.0.1:6000] connections=128, rb=[0,0,10,10,64,64]"
RE_RPC = re.compile(r'\[([^\]]+)\] connections=(\d+), rb=\[([\d,]+)\]')

RB_LABELS = ['start', 'reaped', 'processed', 'streamed', 'assembled', 'end']


class Report:
    def __init__(self):
        self.failed = False

    def ok(self, msg):
        print(f"  ok    {msg}")

    def info(self, msg):
        print(f"        {msg}")

    def fail(self, msg):
        print(f"  FAIL  {msg}")
        self.failed = True

    def warn(self, msg):
        print(f"  warn  {msg}")


def read_lines(path):
    with open(path, errors='replace') as f:
        return f.read().splitlines()


def check_server(lines, r):
    rows = [(int(m.group(1)), int(m.group(2)))
            for m in (RE_SERVER.search(ln) for ln in lines) if m]
    if not rows:
        r.fail("no 'FrbServer: beamset=..., ichunk=...' lines -- format changed?")
        return
    r.ok(f"{len(rows)} per-chunk lines")

    per_beamset = {}
    for beamset, ichunk in rows:
        per_beamset.setdefault(beamset, []).append(ichunk)
    for beamset, seq in sorted(per_beamset.items()):
        bad = sum(1 for i in range(1, len(seq)) if seq[i] < seq[i - 1])
        if bad:
            r.fail(f"beamset {beamset}: ichunk goes backwards {bad} time(s)")
        else:
            r.ok(f"beamset {beamset}: ichunk {seq[0]} -> {seq[-1]}, monotonic")

    # The first per-chunk line has been truncated by an output bug before.
    first = next(ln for ln in lines if RE_SERVER.search(ln))
    if not first.rstrip().endswith(']') and 'beams=' not in first:
        r.fail(f"first per-chunk line looks truncated: {first!r}")
    else:
        r.ok("first per-chunk line is well-formed")


def check_grouper(lines, r):
    base, spikes = [], 0
    for ln in lines:
        m = RE_GROUPER.search(ln)
        if not m:
            continue
        snr, nev = float(m.group(1)), int(m.group(2))
        if nev == 0:
            base.append(snr)
        else:
            spikes += 1
    if not base and not spikes:
        r.fail("no 'coarse_snr_max=..., nevents=...' lines -- format changed?")
        return
    if base:
        r.ok(f"baseline (nevents=0): {len(base)} chunks, "
             f"coarse_snr_max {min(base):.3g} .. {max(base):.3g}, "
             f"median {statistics.median(base):.3g}")
        if max(base) > 20:
            r.warn(f"baseline max {max(base):.3g} is high for a no-event chunk")
    else:
        r.warn("every chunk had events -- no baseline to measure")
    r.ok(f"{spikes} chunk(s) with nevents >= 1")
    if spikes == 0:
        r.warn("no above-threshold events at all (expected if -f was omitted)")


def check_xengine(lines, r):
    ev = [(int(m.group(1)), float(m.group(2)), float(m.group(4)))
          for m in (RE_INJECT.search(ln) for ln in lines) if m]
    if not ev:
        r.warn("no 'injected FRB:' lines (expected if -f was omitted)")
        return
    snrs = sorted({e[2] for e in ev})
    beams = {}
    for b, _, _ in ev:
        beams[b] = beams.get(b, 0) + 1
    dms = [e[1] for e in ev]
    r.ok(f"{len(ev)} injected FRBs across {len(beams)} beams")
    r.info(f"snr values: {snrs}")
    r.info(f"dm range: {min(dms):.4g} .. {max(dms):.4g}")
    per_beam = sorted(beams.values())
    r.info(f"per-beam injection counts: min={per_beam[0]} max={per_beam[-1]}")


def check_sifter(lines, r):
    truth, search = {}, []
    cur = None
    for ln in lines:
        h = RE_SIFTER_HDR.match(ln)
        if h:
            cur = 'truth' if h.group(1) else 'search'
            continue
        m = RE_SIFTER_EV.match(ln)
        if not m or cur is None:
            continue
        beam, dm, fpga, snr = (int(m.group(1)), float(m.group(2)),
                               int(m.group(3)), float(m.group(4)))
        if cur == 'truth':
            truth.setdefault(beam, []).append((fpga, dm))
        else:
            search.append((beam, dm, fpga, snr))

    n_truth = sum(len(v) for v in truth.values())
    if n_truth == 0 and not search:
        r.fail("no sifter event lines at all -- format changed?")
        return
    if n_truth == 0:
        r.fail("no FROM_SIMULATOR (truth) events -- was the fake X-engine "
               "given -s SIFTER_ADDR?")
    else:
        r.ok(f"truth stream: {n_truth} events across {len(truth)} beams")
    if not search:
        r.fail("no search events from the grouper")
        return
    snrs = [s[3] for s in search]
    r.ok(f"search stream: {len(search)} events, snr "
         f"{min(snrs):.3g} .. {max(snrs):.3g}, median {statistics.median(snrs):.3g}")

    if not truth:
        return
    close = 0
    for beam, dm, fpga, _ in search:
        cand = truth.get(beam)
        if not cand:
            continue
        _, tdm = min(cand, key=lambda c: abs(c[0] - fpga))
        if tdm > 0 and 0.5 <= dm / tdm <= 2.0:
            close += 1
    frac = 100.0 * close / len(search)
    msg = (f"{close}/{len(search)} = {frac:.1f}% of search detections have a DM "
           f"within 2x of the nearest same-beam truth injection")
    (r.ok if frac >= 80.0 else r.fail)(msg)


def check_rpc_status(lines, r, acqdir):
    per_addr = {}
    for ln in lines:
        m = RE_RPC.search(ln)
        if m:
            vals = [int(x) for x in m.group(3).split(',')]
            per_addr.setdefault(m.group(1), []).append((int(m.group(2)), vals))
    if not per_addr:
        r.fail("no 'connections=..., rb=[...]' lines -- format changed?")
    for addr, rows in sorted(per_addr.items()):
        bad = []
        ncounters = len(rows[0][1])
        for j in range(ncounters):
            seq = [row[1][j] for row in rows]
            n = sum(1 for i in range(1, len(seq)) if seq[i] < seq[i - 1])
            if n:
                label = RB_LABELS[j] if j < len(RB_LABELS) else f"[{j}]"
                bad.append(f"{label} ({n} backward step(s))")
        conns = sorted({row[0] for row in rows})
        if bad:
            r.fail(f"{addr}: non-monotonic ring-buffer counters: {', '.join(bad)}")
        else:
            r.ok(f"{addr}: {len(rows)} samples, all {ncounters} counters "
                 f"monotonic, connections={conns}")

    recv = sum(1 for ln in lines if ln.rstrip().endswith('received')
               or ' received (stream ' in ln)
    r.ok(f"{recv} file(s) reported received")
    if acqdir:
        n = sum(1 for ln in lines if acqdir in ln and 'received' in ln)
        (r.ok if n else r.fail)(f"{n} received filename(s) mention acqdir {acqdir!r}")


def check_errors(name, lines, r, cascade):
    hits = [(i, ln) for i, ln in enumerate(lines)
            if ERR_RE.search(ln) and not ERR_SKIP_RE.search(ln)]
    if not hits:
        r.ok(f"{name}: no errors ({len(lines)} lines)")
        return
    first = hits[0][0]
    tail_start = max(0, len(lines) - 60)
    if cascade and first >= tail_start:
        r.ok(f"{name}: {len(hits)} error line(s), all in the final "
             f"{len(lines) - first} lines (the expected shutdown cascade)")
    elif cascade:
        r.fail(f"{name}: error at line {first + 1}/{len(lines)}, well before the "
               f"cascade region -- not explained by the deliberate SIGINT")
        r.info(f"first: {hits[0][1].strip()[:120]}")
    else:
        r.fail(f"{name}: {len(hits)} error line(s), first at line {first + 1}")
        r.info(f"first: {hits[0][1].strip()[:120]}")


CHECKERS = {
    'server': check_server,
    'grouper': check_grouper,
    'xengine': check_xengine,
    'sifter': check_sifter,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logdir', required=True)
    ap.add_argument('--cascade', action='store_true',
                    help='the run ended with a deliberate SIGINT, so errors in '
                         'the tail of each log are expected')
    ap.add_argument('--acqdir', metavar='NAME',
                    help='stream acqdir name; check its filenames were reported '
                         'by rpc_status')
    args = ap.parse_args()

    if not os.path.isdir(args.logdir):
        print(f"check-logs: no such directory: {args.logdir}", file=sys.stderr)
        return 2

    r = Report()
    for name in ['server', 'grouper', 'xengine', 'sifter', 'rpc_status']:
        path = os.path.join(args.logdir, f'{name}.log')
        print(f"\n=== {name}.log ===")
        if not os.path.exists(path):
            r.warn(f"missing ({path}) -- skipped")
            continue
        lines = read_lines(path)
        if not lines:
            r.fail("log is empty")
            continue
        if name == 'rpc_status':
            check_rpc_status(lines, r, args.acqdir)
        else:
            CHECKERS[name](lines, r)
        check_errors(name, lines, r, args.cascade)

    print("\n" + ("RESULT: FAIL" if r.failed else "RESULT: PASS"))
    return 1 if r.failed else 0


if __name__ == '__main__':
    sys.exit(main())
