#!/usr/bin/env python3
"""Convert a legacy rf_pipelines json RFI chain to the pirate_frb.chimefrb yaml format, and
write the yaml to stdout.

    legacy_json_to_yaml.py [JSON_FILE] [--nbeams N] [--nfreq F] [--ntime T]

The default json file is the CHIME production chain next to this script,
21-03-07-low-latency-uniform-badchannel-mask-noplot.json, and the default geometry is
1 x 16384 x 4096: one beam, the full CHIME band, and 4096 time samples, the smallest block
at which that chain runs (its clippers have nt_chunk = 4096). The geometry is needed to
BUILD the chain -- every transform is constructed for a block shape -- but the yaml written
records no geometry, and can be read back at any shape the transforms accept.

Notes from the conversion (the elements skipped because they have no pirate counterpart,
the CHIME band assumed for badchannel_mask) go to stderr, so that stdout is clean yaml.
"""

import argparse
import json
import os
import sys

from pirate_frb.chimefrb import PIPELINE_YAML_HEADER, transform_from_json_dict, yaml_string

DEFAULT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '21-03-07-low-latency-uniform-badchannel-mask-noplot.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('json_file', nargs='?', default=DEFAULT_JSON)
    parser.add_argument('--nbeams', type=int, default=1)
    parser.add_argument('--nfreq', type=int, default=16384)
    parser.add_argument('--ntime', type=int, default=4096)
    args = parser.parse_args()

    with open(args.json_file) as f:
        d = json.load(f)

    chain = transform_from_json_dict(d, args.nbeams, args.nfreq, args.ntime)
    if chain is None:
        sys.exit(f'{args.json_file}: the top-level element has no pirate counterpart')

    header = (f'# Converted by misc/chimefrb/configs/legacy_json_to_yaml.py from\n'
              f'# {os.path.basename(args.json_file)} at nbeams={args.nbeams}'
              f' nfreq={args.nfreq} ntime={args.ntime}.\n' + PIPELINE_YAML_HEADER)
    sys.stdout.write(yaml_string(chain.to_yaml_dict(), header=header))


if __name__ == '__main__':
    main()
