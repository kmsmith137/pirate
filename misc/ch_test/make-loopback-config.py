#!/usr/bin/env python3
"""Rewrite a production frb_server config's data addresses to loopback.

The production config names the node's physical 10.x data NICs, which are not
visible inside the sandbox (private network namespace). This copies the config
to a NEW file and rewrites every data_ip_addrs entry to 127.0.0.1 with a
unique port -- all receivers now share one IP, so they can no longer share a
port number.

    make-loopback-config.py configs/frb_server/cf05_production.yml OUT.yml

The rewrite is line-based, not a yaml round-trip: everything outside the
rewritten address lines is preserved byte-for-byte, including comments and
formatting. Nothing else changes -- memory sizes, dedispersion config, ssd/nfs
dirs, check_mountpoints and MTU minimums are all left alone (loopback's MTU
65536 clears min_data_mtu on its own).

The output must be an UNTRACKED path (the tracked config is shared with the
real cluster and must not be edited): use the session scratch dir, /tmp, or
plans/. The script refuses to overwrite the input.

rpc_ip_addrs is left alone by default: its '10.222.3.*' glob usually resolves
inside the sandbox, since the sandbox mirrors the host's default interface.
Pass --rpc-loopback if preflight-prod.py reports that it does not, and then
use the loopback addresses in every rpc_* command.

Prints the diff it made, so the caller can confirm only the intended lines
changed.
"""

import argparse
import os
import re
import sys

# "  - [ '10.0.0.*:5000', '10.0.1.*:5000' ]" or "  - '10.222.3.*:6000'"
RE_ADDR = re.compile(r"'([^']*:\d+)'")


def rewrite_block(lines, key, first_port, host='127.0.0.1'):
    """Rewrite quoted ip:port tokens inside the 'key:' block. Returns
    (new_lines, [(lineno, old, new)], next_port)."""
    out, changes = [], []
    port = first_port
    in_block = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(rf'^{re.escape(key)}\s*:', stripped):
            in_block = True
            out.append(line)
            continue
        if in_block:
            # The block ends at the first line that is neither a list item nor
            # a blank/comment line.
            if stripped.startswith('-'):
                def sub(m):
                    nonlocal port
                    new = f"{host}:{port}"
                    port += 1
                    return f"'{new}'"
                new_line = RE_ADDR.sub(sub, line)
                if new_line != line:
                    changes.append((i + 1, line.rstrip('\n'), new_line.rstrip('\n')))
                out.append(new_line)
                continue
            if stripped == '' or stripped.startswith('#'):
                out.append(line)
                continue
            in_block = False
        out.append(line)
    return out, changes, port


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', help='the TRACKED production config (read-only)')
    ap.add_argument('dst', help='output path (must be untracked scratch)')
    ap.add_argument('--first-data-port', type=int, default=5000)
    ap.add_argument('--rpc-loopback', action='store_true',
                    help="also rewrite rpc_ip_addrs to loopback (only if its "
                         "glob does not resolve in this sandbox)")
    ap.add_argument('--first-rpc-port', type=int, default=6000)
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        print("make-loopback-config: refusing to overwrite the input "
              "(the tracked config must not be edited)", file=sys.stderr)
        return 2

    with open(args.src) as f:
        lines = f.read().splitlines(keepends=True)

    lines, changes, _ = rewrite_block(lines, 'data_ip_addrs', args.first_data_port)
    if not changes:
        print(f"make-loopback-config: no data_ip_addrs entries rewritten in "
              f"{args.src} -- wrong config, or the format changed?", file=sys.stderr)
        return 2

    if args.rpc_loopback:
        lines, rpc_changes, _ = rewrite_block(lines, 'rpc_ip_addrs',
                                              args.first_rpc_port)
        changes += rpc_changes

    with open(args.dst, 'w') as f:
        f.writelines(lines)

    print(f"wrote {args.dst} ({len(changes)} line(s) changed):")
    for lineno, old, new in changes:
        print(f"  line {lineno}:")
        print(f"    - {old.strip()}")
        print(f"    + {new.strip()}")
    print("\nEverything else is byte-identical to the source. Pass the new file "
          "to run_server in place of the tracked one.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
