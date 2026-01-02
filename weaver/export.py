import os
import sys


if __name__=='__main__':

    # read model directory with all info
    modeldir = sys.argv[1]

    # find all required files
    dataconfig = os.path.join(modeldir, 'data_config.yaml')
    modelconfig = os.path.join(modeldir, 'model_config.py')
    modelstate = os.path.join(modeldir, 'network_best_epoch_state.pt')
    tocheck = [dataconfig, modelconfig, modelstate]
    for f in tocheck:
        if not os.path.exists(f):
            msg = f'Expected file {f} does not exist.'
            raise Exception(msg)

    # set output
    outputname = 'model.onnx'

    # make command
    cmd = 'weaver'
    cmd += f' -c {dataconfig}'
    cmd += f' -n {modelconfig}'
    cmd += f' -m {modelstate}'
    cmd += f' --export-onnx {outputname}'

    # run command
    os.system(cmd)
