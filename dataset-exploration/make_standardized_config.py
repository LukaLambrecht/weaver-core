# Make a data config with fixed standardization constants

# Takes a data config in which the variables to standardize are marked as [<variable>, auto],
# calculates the standardization constants (center and scale) on a given set of files,
# and writes a copy of the data config with [<variable>, <center>, <scale>] instead.
# The constants are calculated in the same way as weaver itself would do
# at the start of the training (center = median, scale = 1 / 68% half-width),
# except that default values (-9, e.g. track variables for neutral particles) are ignored;
# this reproduces the constants in the existing standardized configs.

import os
import re
import sys
import glob
import argparse
import numpy as np
import awkward as ak

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(weavercoredir)
from weaver.utils.data.config import DataConfig
from weaver.utils.data.preprocess import AutoStandardizer
from weaver.utils.samplelisttools import read_sample_list


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
      help='Input data config (with variables to standardize marked as "auto").')
    parser.add_argument('-s', '--samplelist', required=True,
      help='Sample list (.yaml) with files on which to calculate the constants.')
    parser.add_argument('-o', '--outputfile', required=True,
      help='Output data config.')
    parser.add_argument('-f', '--data_fraction', default=None, type=float,
      help='Fraction of entries to read from each file (default: as specified in the data config).')
    parser.add_argument('-d', '--default_value', default=-9., type=float,
      help='Default value to ignore in the calculation.')
    args = parser.parse_args()

    # find input files
    files = []
    for pattern in read_sample_list(args.samplelist): files += sorted(glob.glob(pattern))
    print(f'Found {len(files)} input files.')
    if len(files)==0: raise Exception('No input files found.')

    # calculate the standardization constants
    data_config = DataConfig.load(args.config)
    if args.data_fraction is not None:
        data_config.options['preprocess']['data_fraction'] = args.data_fraction
    table = AutoStandardizer(files, data_config).read_file(files)
    print(f'Using {len(table)} jets to calculate the standardization constants.')
    params = {}
    for variable in table.fields:
        values = ak.to_numpy(ak.flatten(table[variable], axis=None)).astype(float)
        values = values[np.isfinite(values) & (values != args.default_value)]
        (low, center, high) = np.percentile(values, [16, 50, 84])
        scale = max(high - center, center - low)
        scale = 1. if scale==0 else 1./scale
        params[variable] = {'center': center, 'scale': scale}
        print(f'  - {variable}: center = {center:.6g}, scale = {scale:.6g} ({len(values)} values)')

    # write the output config
    # (by text replacement, in order to keep the formatting and comments of the input config)
    with open(args.config) as f: config = f.read()
    def replace(match):
        variable = match.group(1)
        return f'- [{variable}, {params[variable]["center"]:.6g}, {params[variable]["scale"]:.6g}]'
    (config, nreplaced) = re.subn(r'- \[(\w+), auto\]', replace, config)
    with open(args.outputfile, 'w') as f: f.write(config)
    print(f'Replaced {nreplaced} entries, output written to {args.outputfile}.')
