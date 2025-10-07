import os
import sys
import json
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(weavercoredir)
import weaver.utils.jobsubmission.condortools as ct
import weaver.utils.jobsubmission.slurmtools as st


if __name__=='__main__':

    # common settings
    weaverdir = os.path.join(weavercoredir, 'weaver')
    # data config
    data_config = os.path.abspath('configs/data_config_pnet.yaml')
    #data_config = os.path.abspath('configs/data_config_part.yaml')
    # model config
    model_config = os.path.abspath('configs/model_config_pnet.py')
    #model_config = os.path.abspath('configs/model_config_part.py')
    # sample list for training data
    sample_config_train = os.path.abspath('configs/samplelists/uflhpg/samples_training_smalltest.yaml')
    # sample list for testing data
    sample_config_test = os.path.abspath('configs/samplelists/uflhpg/samples_testing_smalltest.yaml')
    # output dir
    outputdir = os.path.join(thisdir, 'output_test')
    # network settings
    num_epochs = 10
    steps_per_epoch = 30
    batch_size = 512
    # runmode and job settings
    # (choose from 'local', 'condor', and 'slurm')
    runmode = 'local'
    gpus= '0'

    # check if all config files exist
    files_to_check = [data_config, model_config, sample_config_train, sample_config_test]
    for f in files_to_check:
        if not os.path.exists(f):
            raise Exception('File {} does not exist.'.format(f))

    # make output directory
    if os.path.exists(outputdir):
        msg = f'Output directory {outputdir} already exists.'
        raise Exception(msg)
    os.makedirs(outputdir)

    # copy the config files to the output directory
    this_data_config = os.path.join(outputdir, 'data_config.yaml')
    os.system(f'cp {data_config} {this_data_config}')
    this_model_config = os.path.join(outputdir, 'model_config.py')
    os.system(f'cp {model_config} {this_model_config}')
    this_sample_config_train = os.path.join(outputdir, 'sample_config_train.yaml')
    os.system(f'cp {sample_config_train} {this_sample_config_train}')
    this_sample_config_test = os.path.join(outputdir, 'sample_config_test.yaml')
    os.system(f'cp {sample_config_test} {this_sample_config_test}')

    # set model prefix
    model_prefix = os.path.join(outputdir, 'network')

    # set output file for test results
    test_output = os.path.join(outputdir, 'output.root')

    # make the command
    cmd = 'weaver'
    cmd += f' --data-train {this_sample_config_train}'
    cmd += f' --data-config {this_data_config}'
    cmd += f' --network-config {this_model_config}'
    cmd += f' --num-epochs {num_epochs}'
    cmd += f' --steps-per-epoch {steps_per_epoch}'
    cmd += f' --batch-size {batch_size}'
    cmd += f' --model-prefix {model_prefix}'
    cmd += f' --data-test {this_sample_config_test}'
    cmd += f' --predict-output {test_output}'
    # data loading options
    cmd += ' --in-memory --fetch-step 1'
    cmd += ' --copy-inputs'
    # compute options
    if gpus is not None: cmd += f' --gpus {gpus}'

    # run or submit commands
    if runmode == 'local':
        print(cmd)
        os.system(cmd)
    elif runmode=='condor':
        conda_activate = 'source /eos/user/l/llambrec/miniforge3/bin/activate'
        conda_env = 'weaver'
        ct.submitCommandAsCondorJob('cjob_weaver', cmd,
          jobflavour='workday', conda_activate=conda_activate, conda_env=conda_env)
    elif runmode=='slurm':
        slurmscript = 'sjob_weaver.sh'
        env_cmds = ([
          'source /eos/user/l/llambrec/miniforge3/bin/activate',
          'conda activate weaver',
          f'cd {thisdir}'
        ])
        job_name = os.path.splitext(slurmscript)[0]
        slurm_options = {
          'job_name': job_name,
          'env_cmds': env_cmds,
          'memory': '16G',
          'time': '05:00:00',
          'constraint': 'el9'
        }
        if gpus!='""':
            slurm_options['gres'] = 'gpu:1'
            slurm_options['gpus'] = '1'
        st.submitCommandAsSlurmJob(cmd, slurmscript, **slurm_options)
