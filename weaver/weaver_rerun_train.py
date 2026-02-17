# Simple utility script to re-run the training of a given model.


import os
import sys

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(weavercoredir)
import weaver.utils.jobsubmission.condortools as ct
import weaver.utils.jobsubmission.slurmtools as st


if __name__=='__main__':

    # read model directory with all info
    modeldir = sys.argv[1]

    # network settings
    num_epochs = 50
    steps_per_epoch = 300
    batch_size = 512
    # runmode and job settings
    # (choose from 'local', 'condor', and 'slurm')
    runmode = 'slurm'
    gpus= '0'

    # settings based on user
    user = os.getenv('USER')
    miniforge = None
    if user=='llambre1.brown':
        miniforge = '/blue/avery/llambre1.brown/miniforge3/bin/activate'
    if user=='llambrec':
        miniforge = '/eos/user/l/llambrec/miniforge3/bin/activate'
    if user=='tgillin':
        miniforge = '/users/tgillin/miniconda3/etc/profile.d/conda.sh'

    # find all required files
    train_samples = os.path.join(modeldir, 'sample_config_train.yaml')
    test_samples = os.path.join(modeldir, 'sample_config_test.yaml')
    dataconfig = os.path.join(modeldir, 'data_config.yaml')
    modelconfig = os.path.join(modeldir, 'model_config.py')
    tocheck = [train_samples, test_samples, dataconfig, modelconfig]
    for f in tocheck:
        if f is None: continue
        if not os.path.exists(f):
            msg = f'Expected file {f} does not exist.'
            raise Exception(msg)

    # set model prefix
    model_prefix = os.path.join(modeldir, 'network')

    # set output
    outputfile = os.path.join(modeldir, 'output_rerun_train.root')
    if os.path.exists(outputfile):
        msg = f'Output file {outputfile} already exists.'
        raise Exception(msg)

    # make the command
    cmd = 'python train.py'
    cmd += f' --data-train {train_samples}'
    cmd += f' --data-config {dataconfig}'
    cmd += f' --network-config {modelconfig}'
    cmd += f' --num-epochs {num_epochs}'
    cmd += f' --steps-per-epoch {steps_per_epoch}'
    cmd += f' --batch-size {batch_size}'
    cmd += f' --model-prefix {model_prefix}'
    cmd += f' --data-test {test_samples}'
    cmd += f' --predict-output {outputfile}'
    # data loading options
    cmd += ' --num-workers 6'
    #cmd += ' --in-memory --fetch-step 1'
    cmd += ' --copy-inputs'
    # compute options
    if gpus is not None: cmd += f' --gpus {gpus}'

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
