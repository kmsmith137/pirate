#!/usr/bin/env python3

import json

def summarize_rfi_json(x):
    if isinstance(x, list):
        for y in x:
            summarize_rfi_json(y)
        return
    
    assert isinstance(x, dict)
    assert 'class_name' in x
    cname = x['class_name']

    if cname == 'pipeline':
        summarize_rfi_json(x['elements'])
    elif cname == 'wi_sub_pipeline':
        summarize_rfi_json(x['sub_pipeline'])
    elif cname in ['std_dev_clipper','polynomial_detrender','spline_detrender']:
        print(f'    Skipping {cname} for now')
    elif cname == 'intensity_clipper':
        Df, Dt, axis, nt_chunk, niter = x['Df'], x['Dt'], x['axis'], x['nt_chunk'], x['niter']
        if axis == 'AXIS_NONE':
            Cf, Ct = 1024//Df, nt_chunk//Dt
        elif axis == 'AXIS_FREQ':
            Cf, Ct = 1024//Df, 1
        elif axis == 'AXIS_TIME':
            Cf, Ct = 1, x['nt_chunk']//Dt
        else:
            raise RuntimeError(f'unrecognized {axis=}')
        print(f'intensity_clipper: {Df=} {Dt=} {Cf=} {Ct=} {niter=} Dtot={Df*Dt} Ctot={Cf*Ct}')
    elif cname not in ['mask_counter','badchannel_mask']:
        raise RuntimeError(f'{cname=} keys={list(x.keys())}')


if __name__ == '__main__':
    filename = 'rfi.json'
    summarize_rfi_json(json.load(open(filename)))
