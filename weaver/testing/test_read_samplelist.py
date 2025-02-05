import os
import sys
import argparse

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(weavercoredir)
import weaver.utils.samplelisttools as samplelisttools


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--samplelist', required=True)
    args = parser.parse_args()

    # read sample list and print results
    samples = samplelisttools.read_sample_list(args.samplelist)
    print(samples)
