import os
import sys
import json
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(weavercoredir)
import weaver.utils.jobsubmission.condortools as ct
import weaver.utils.jobsubmission.slurmtools as st


if __name__=='__main__':

    # common settings
    weaverdir = os.path.join(weavercoredir, 'weaver')
    # data config
    data_config = os.path.abspath('configs/data_config_part_disco.yaml')
    # model config
    #model_config = os.path.abspath('../disco-withsyndata-feynnetpairing/results_20250725/output_part/alpha_5p00/training_0/model_config.py')
    model_config = os.path.abspath('../disco-withsyndata/results_20250522/output_part/alpha_10p00/training_0/model_config.py')
    # model weights
    #model_weights = os.path.abspath('../disco-withsyndata-feynnetpairing/results_20250725/output_part/alpha_5p00/training_0/network_best_epoch_state.pt')
    model_weights = os.path.abspath('../disco-withsyndata/results_20250522/output_part/alpha_10p00/training_0/network_best_epoch_state.pt')
    # sample list
    sample_config = os.path.abspath('configs/samples_testing.yaml')
    # output dir
    outputdir = os.path.join(thisdir, 'output_test_old2')
    # runmode and job settings
    runmode = 'slurm'
    #conda_activate = 'source /eos/user/l/llambrec/miniforge3/bin/activate'
    #conda_env = 'weaver'
    slurmscript = 'sjob_weaver_test_old2.sh'
    env_cmds = ([
        'env_path=/blue/avery/llambre1.brown/miniforge3/envs/weaver/bin/',
        'export PATH=$env_path:$PATH',
        f'cd {thisdir}'
    ])
    gpus = '0'

    # network settings
    batch_size = 512

    # check if all config files exist
    files_to_check = [data_config, model_config, model_weights, sample_config]
    for f in files_to_check:
        if not os.path.exists(f):
            raise Exception('File {} does not exist.'.format(f))

    # make output directory
    os.makedirs(outputdir)

    # copy the config files to the working directory
    # (note: this is especially needed for the data config file,
    #  as weaver writes an auto-generated config file,
    #  which can give issues with many parallel jobs.
    #  alternatively, the --no-remake-weights option could be used,
    #  but copying the config files is probably cleaner.)
    this_data_config = os.path.join(outputdir, 'data_config.yaml')
    os.system(f'cp {data_config} {this_data_config}')
    this_model_config = os.path.join(outputdir, 'model_config.py')
    os.system(f'cp {model_config} {this_model_config}')
    this_model_weights = os.path.join(outputdir, 'model_weights.pt')
    os.system(f'cp {model_weights} {this_model_weights}')
    this_sample_config = os.path.join(outputdir, 'sample_config.yaml')
    os.system(f'cp {sample_config} {this_sample_config}')

    # set output file for test results
    test_output = os.path.join(outputdir, 'output.root')

    # make the command
    cmd = 'weaver --predict'
    cmd += f' --data-config {this_data_config}'
    cmd += f' --network-config {this_model_config}'
    cmd += f' --model-prefix {model_weights}'
    cmd += f' --batch-size {batch_size}'
    cmd += f' --data-test {this_sample_config}'
    cmd += f' --predict-output {test_output}'
    if gpus is not None: cmd += f' --gpus {gpus}'
    # data loading options
    #cmd += ' --in-memory --fetch-step 1'
    cmd += ' --copy-inputs'

    # run or submit commands
    if runmode == 'local':
        print(cmd)
        os.system(cmd)
    elif runmode=='condor':
        ct.submitCommandAsCondorJob('cjob_weaver', cmd,
          jobflavour='workday', conda_activate=conda_activate, conda_env=conda_env)
    elif runmode=='slurm':
        job_name = os.path.splitext(slurmscript)[0]
        slurm_options = {
          'job_name': job_name,
          'env_cmds': env_cmds,
          'memory': '32G',
          'time': '05:00:00',
          'constraint': 'el9'
        }
        if gpus is not None and gpus!='""':
            slurm_options['gres'] = 'gpu:1'
            slurm_options['gpus'] = '1'
        st.submitCommandAsSlurmJob(cmd, slurmscript, **slurm_options)
