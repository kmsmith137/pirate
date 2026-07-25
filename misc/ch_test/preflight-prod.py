#!/usr/bin/env python3
"""Check that this machine can host a production run, BEFORE starting one.

The production frb_server config assumes a real CHORD FRB node: NFS and
scratch mounts, a 1.5 TiB hugepage pool, both GPUs idle, and the node's
10.x data NICs. Inside a sandbox some of those may be missing, and the
failure mode without this check is a confusing crash a minute into server
init -- after the agent has already torn down whatever else was running.

    preflight-prod.py configs/frb_server/cf05_production.yml

Checks, all read-only:

  - $USER is set (the nfs_dir '{user}' interpolation needs it)
  - every check_mountpoints entry is a real mountpoint (os.path.ismount)
  - every ssd_dir exists and is writable
  - nfs_dir resolves and is writable (or its parent is, if it must be created)
  - free hugepages >= num_servers * host_memory_per_server, when
    use_hugepages is set
  - all GPUs are visible and idle (nvidia-smi)
  - every rpc_ip_addrs entry resolves against a local address, and the
    resolved address is exempt from the egress proxy via $NO_PROXY
  - loopback MTU >= min_data_mtu (data_ip_addrs get rewritten to loopback by
    make-loopback-config.py, and loopback's 65536 clears the jumbo minimum)

Exit status 0 if everything needed is present, 1 otherwise. A failure here is
usually NOT something to work around: the sandbox can only be changed from
outside, so report it and ask.
"""

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys

import yaml

UNITS = {'b': 1, 'kib': 1024, 'mib': 1024**2, 'gib': 1024**3, 'tib': 1024**4,
         'kb': 1000, 'mb': 1000**2, 'gb': 1000**3, 'tb': 1000**4}


def parse_bytes(s):
    """'768 GiB' -> 824633720832. Returns None if unparseable."""
    if isinstance(s, (int, float)):
        return int(s)
    m = re.fullmatch(r'\s*([\d.]+)\s*([a-zA-Z]*)\s*', str(s))
    if not m:
        return None
    val, unit = float(m.group(1)), (m.group(2) or 'b').lower()
    return int(val * UNITS[unit]) if unit in UNITS else None


def interpolate(path):
    return (str(path)
            .replace('{user}', os.environ.get('USER', ''))
            .replace('{home}', os.path.expanduser('~')))


def local_ipv4_addrs():
    out = subprocess.run(['ip', '-4', '-o', 'addr', 'show'],
                         capture_output=True, text=True).stdout
    return re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', out)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='production frb_server yaml (the TRACKED one)')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    r = Report()
    print(f"=== {args.config} ===")

    user = os.environ.get('USER')
    (r.ok if user else r.fail)(f"$USER = {user!r}")

    print("\n=== mountpoints ===")
    for d in cfg.get('check_mountpoints') or []:
        d = interpolate(d)
        if not os.path.exists(d):
            r.fail(f"{d}: does not exist")
        elif not os.path.ismount(d):
            r.fail(f"{d}: exists but is NOT a mountpoint "
                   f"(sandbox launched without this mount?)")
        else:
            r.ok(f"{d}: mountpoint")

    print("\n=== ssd dirs ===")
    for d in cfg.get('ssd_dirs') or []:
        d = interpolate(d)
        if not os.path.isdir(d):
            r.fail(f"{d}: not a directory")
        elif not os.access(d, os.W_OK):
            r.fail(f"{d}: not writable")
        else:
            r.ok(f"{d}: exists, writable")

    print("\n=== nfs dir ===")
    nfs = interpolate(cfg.get('nfs_dir', ''))
    if not nfs:
        r.fail("nfs_dir is empty")
    elif os.path.isdir(nfs):
        (r.ok if os.access(nfs, os.W_OK) else r.fail)(
            f"{nfs}: exists, writable={os.access(nfs, os.W_OK)}")
    else:
        parent = os.path.dirname(nfs.rstrip('/'))
        if os.access(parent, os.W_OK):
            r.ok(f"{nfs}: missing but parent {parent} is writable (will be created)")
        else:
            r.fail(f"{nfs}: missing and parent {parent} is not writable")

    print("\n=== hugepages ===")
    if cfg.get('use_hugepages'):
        mi = {}
        with open('/proc/meminfo') as f:
            for ln in f:
                k, _, v = ln.partition(':')
                mi[k] = v.strip()
        psz = parse_bytes(mi.get('Hugepagesize', '2048 kB').replace('kB', 'KiB'))
        free = int(mi.get('HugePages_Free', '0').split()[0])
        total = int(mi.get('HugePages_Total', '0').split()[0])
        per = parse_bytes(cfg.get('host_memory_per_server', 0))
        n = int(cfg.get('num_servers', 1))
        if per is None:
            r.fail(f"could not parse host_memory_per_server="
                   f"{cfg.get('host_memory_per_server')!r}")
        else:
            need_pages = -(-(per * n) // psz)   # ceil
            msg = (f"need {n} x {per / 2**30:.0f} GiB = {per * n / 2**30:.0f} GiB "
                   f"= {need_pages} pages; free {free}/{total} "
                   f"({free * psz / 2**30:.0f} GiB)")
            (r.ok if free >= need_pages else r.fail)(msg)
            if free < need_pages and free + 0 < total:
                r.info("some hugepages are in use -- is another server still running?")
    else:
        r.ok("use_hugepages is false -- skipped")

    print("\n=== gpus ===")
    if shutil.which('nvidia-smi') is None:
        r.fail("nvidia-smi not found")
    else:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
             '--format=csv,noheader'], capture_output=True, text=True)
        rows = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
        if not rows:
            r.fail("nvidia-smi reported no GPUs")
        for ln in rows:
            parts = [p.strip() for p in ln.split(',')]
            used = int(re.sub(r'[^\d]', '', parts[2]) or 0)
            (r.ok if used < 100 else r.warn)(f"gpu {parts[0]} ({parts[1]}): "
                                             f"{parts[2]} used, {parts[3]} util")

    print("\n=== rpc addresses ===")
    addrs = local_ipv4_addrs()
    r.info(f"local ipv4: {addrs}")
    no_proxy = (os.environ.get('NO_PROXY', '') + ',' +
                os.environ.get('no_proxy', '')).split(',')
    no_proxy = {x.strip() for x in no_proxy if x.strip()}
    for spec in cfg.get('rpc_ip_addrs') or []:
        host, _, port = str(spec).rpartition(':')
        match = [a for a in addrs if fnmatch.fnmatch(a, host)]
        if not match:
            r.warn(f"{spec}: no local address matches '{host}' -- rewrite this "
                   f"entry to loopback and use that in the rpc_* commands")
            continue
        r.ok(f"{spec}: resolves to {match[0]}")
        if match[0] not in no_proxy and '127.0.0.1' not in no_proxy:
            r.fail(f"{match[0]} is not in $NO_PROXY -- rpc calls to the node's own "
                   f"IP will hit the egress proxy and fail with a 403. The sandbox "
                   f"launcher predates the node-local exemption; ask for a relaunch.")

    print("\n=== mtu ===")
    min_mtu = int(cfg.get('min_data_mtu', 0))
    out = subprocess.run(['ip', 'link', 'show', 'lo'],
                         capture_output=True, text=True).stdout
    m = re.search(r'mtu (\d+)', out)
    if not m:
        r.warn("could not read loopback MTU")
    else:
        lo_mtu = int(m.group(1))
        (r.ok if lo_mtu >= min_mtu else r.fail)(
            f"loopback mtu {lo_mtu} vs min_data_mtu {min_mtu}")

    print("\n" + ("RESULT: FAIL -- do NOT start the production run; report what is "
                  "missing and ask (the sandbox can only be changed from outside)"
                  if r.failed else "RESULT: PASS"))
    return 1 if r.failed else 0


if __name__ == '__main__':
    sys.exit(main())
