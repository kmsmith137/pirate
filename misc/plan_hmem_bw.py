#!/usr/bin/env python


def show_plan(dd_yml, nfreq, ts, nbeams, dd_hmem_bw):
    """
    nbeams and dd_hmem_bw are from 'pirate_frb time_dedisperser <dd_yml>'.
    These values are per-NUMA, not per-node!
    """
    
    ssd_bw = 6.5                # assumed throughout
    ssd_hmem_bw = 5 * ssd_bw    # empirical, based on 'pirate_frb hwtest configs/hwtest/cf00_asdf.yml'

    gsamples_per_sec = 1.0e-9 * nbeams * nfreq / ts
    net_gbps = gsamples_per_sec * (1056 / 256)   # 1056 bits per 256 time samples
    net_hmem_bw = 5 * (net_gbps / 8.)            # 5 read-write cycles?
    ds_hmem_bw = (gsamples_per_sec) * (4/8 + 2*5/16 + 2*6/32 + 2*7/64 + 8/128)
    total_hmem_bw = net_hmem_bw + ds_hmem_bw + dd_hmem_bw + ssd_hmem_bw

    print(dd_yml)
    print(f'  {nfreq = }')
    print(f'  {ts = }')
    print(f'  {nbeams = }             # from "pirate_frb time_dedisperser {dd_yml}"')
    print(f'  {net_gbps = :.02f}      # per numa, not per-node or per-NIC')
    print(f'  {net_hmem_bw = :.02f}   # 5 read-write cycles assumed')
    print(f'  {ds_hmem_bw = :.02f}    # assumes telescoping ring buffer')
    print(f'  {dd_hmem_bw = :.02f}    # from "pirate_frb time_dedisperser {dd_yml}"')
    print(f'  {ssd_hmem_bw = :.02f}   # empirical, based on "pirate_frb hwtest configs/hwtest/cf00_asdf.yml"')
    print(f'  {total_hmem_bw = :.02f}')


if __name__ == '__main__':
    print()
    show_plan('configs/dedispersion/chime_sb1.yml', nfreq=20000, ts=1.0e-3, nbeams=320, dd_hmem_bw=37.0)
    print()
    show_plan('configs/dedispersion/chord_sb1.yml', nfreq=40000, ts=1.0e-3, nbeams=150, dd_hmem_bw=38.0)
    print()
