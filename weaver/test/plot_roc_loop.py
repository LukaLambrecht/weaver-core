import os
import sys
import argparse


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--modeldirs', required=True, nargs='+')
    parser.add_argument('-i', '--inputfilename', default='output.root')
    parser.add_argument('-s', '--signal_mask', required=True)
    parser.add_argument('-t', '--treename', default=None)
    parser.add_argument('-c', '--correlations', default=[], nargs='+')
    parser.add_argument('--plot_score_dist', default=False, action='store_true')
    parser.add_argument('--plot_roc', default=False, action='store_true')
    parser.add_argument('--plot_correlation', default=False, action='store_true')
    args = parser.parse_args()

    # loop over model dirs
    for modeldir in args.modeldirs:
        
        # find input file
        inputfile = os.path.join(modeldir, args.inputfilename)
        if not os.path.exists(inputfile):
            print(f'Warning: file {inputfile} does not exist, skipping.')
            continue

        # set output dir
        outputdir = os.path.join(modeldir, 'output_plots')
        
        # make the command
        cmd = 'python3 plot_roc.py'
        cmd += f' -i {inputfile}'
        cmd += f' -o {outputdir}'
        cmd += f' -s {args.signal_mask}'
        if args.treename is not None: cmd += f' -t {args.treename}'
        if len(args.correlations)>0: cmd += ' -c {}'.format(' '.join(args.correlations))
        if args.plot_score_dist: cmd += ' --plot_score_dist'
        if args.plot_roc: cmd += ' --plot_roc'
        if args.plot_correlation: cmd += ' --plot_correlation'

        # run the command
        os.system(cmd)
