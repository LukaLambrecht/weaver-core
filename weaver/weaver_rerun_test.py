# Simple utility script to re-run the evaluation of a given model.


import os
import sys
import numpy as np
from fnmatch import fnmatch

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(weavercoredir)
import weaver.utils.jobsubmission.condortools as ct
import weaver.utils.jobsubmission.slurmtools as st


if __name__=='__main__':

    # read model directory with all info
    modeldir = sys.argv[1]

    # hard-coded settings (maybe add as argument later)
    which_model = 'best' # choose from "best", "onnx" or "latest"
    runmode = 'slurm'
    gpus= '0'

    # find all required files
    samples = os.path.join(modeldir, 'sample_config_test.yaml')
    dataconfig = os.path.join(modeldir, 'data_config.yaml')
    modelconfig = os.path.join(modeldir, 'model_config.py')
    if which_model=='onnx': modelstate = os.path.join(modeldir, 'model.onnx')
    elif which_model=='best': modelstate = os.path.join(modeldir, 'network_best_epoch_state.pt')
    elif which_model=='latest':
        candidates = [f for f in os.listdir(modeldir) if fnmatch(f, 'network_epoch-*_state.pt')]
        epoch_numbers = [int(f.split('epoch-')[-1].replace('_state.pt', '')) for f in candidates]
        idx = np.argmax(epoch_numbers)
        modelstate = os.path.join(modeldir, candidates[idx])
    tocheck = [samples, dataconfig, modelconfig, modelstate]
    for f in tocheck:
        if f is None: continue
        if not os.path.exists(f):
            msg = f'Expected file {f} does not exist.'
            raise Exception(msg)

    # set output
    outputfile = os.path.join(modeldir, 'output_rerun_test.root')
    if os.path.exists(outputfile):
        msg = f'Output file {outputfile} already exists.'
        raise Exception(msg)

    # make command
    cmd = 'python train.py --predict'
    cmd += f' --data-test {samples}'
    cmd += f' --data-config {dataconfig}'
    cmd += f' --network-config {modelconfig}'
    cmd += f' --model-prefix {modelstate}'
    cmd += f' --gpus 0'
    cmd += f' --batch-size 512'
    cmd += f' --predict-output {outputfile}'

    # settings based on user
    user = os.getenv('USER')
    miniforge = None
    if user=='llambre1.brown':
        miniforge = '/blue/avery/llambre1.brown/miniforge3/bin/activate'
    if user=='llambrec':
        miniforge = '/eos/user/l/llambrec/miniforge3/bin/activate'
    if user=='tgillin':
        miniforge = '/users/tgillin/miniconda3/etc/profile.d/conda.sh'

    # run or submit commands
    if runmode == 'local':
        print(cmd)
        os.system(cmd)
    elif runmode=='condor':
        conda_activate = f'source {miniforge}'
        conda_env = 'weaver'
        ct.submitCommandAsCondorJob('cjob_weaver', cmd,
          jobflavour='workday', conda_activate=conda_activate, conda_env=conda_env)
    elif runmode=='slurm':
        slurmscript = 'sjob_weaver.sh'
        # remove old slurm script if it exists
        if os.path.exists(slurmscript):
            os.remove(slurmscript)
        env_cmds = ([
          f'source {miniforge}',
          'conda activate weaver',
          f'cd {thisdir}'
        ])
        job_name = os.path.splitext(slurmscript)[0]
        slurm_options = {
          'job_name': job_name,
          'env_cmds': env_cmds,
          'memory': '16G',
          'time': '15:00:00'
        }
        if gpus!='""':
            slurm_options['gpus'] = '1'
        st.submitCommandAsSlurmJob(cmd, slurmscript, **slurm_options)
