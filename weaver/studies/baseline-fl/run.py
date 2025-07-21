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
    data_config = os.path.abspath('configs/data_config_hh4b_mh125_vs_bkg_pnet_flatloss.yaml')
    #data_config = os.path.abspath('configs/data_config_hh4b_mh125_vs_bkg_part_flatloss.yaml')
    # model config
    model_config = os.path.abspath('configs/model_pnet_flatloss.py')
    #model_config = os.path.abspath('configs/model_part_flatloss.py')
    # sample list for training data
    sample_config_train = os.path.abspath('configs/samples_hh4b_mh125_vs_bkg_allyears_training.yaml')
    # sample list for testing data
    sample_config_test = os.path.abspath('configs/samples_hh4b_mh125_vs_bkg_allyears_testing.yaml')
    # output dir
    outputdir = os.path.join(thisdir, 'output_pnet')
    # set alpha range
    alphas = [1000]
    # network settings
    num_epochs = 30
    steps_per_epoch = 150
    batch_size = 1024
    num_trainings = 1
    flatloss_label = 0
    # specify whether to run training or only print preparatory steps
    do_training = True
    # runmode and job settings
    runmode = 'slurm'
    #conda_activate = 'source /eos/user/l/llambrec/miniforge3/bin/activate'
    #conda_env = 'weaver'
    slurmscript = 'sjob_weaver_pnet.sh'
    env_cmds = ([
        'env_path=/blue/avery/llambre1.brown/miniforge3/envs/weaver/bin/',
        'export PATH=$env_path:$PATH',
        f'cd {thisdir}'
    ])
    gpus = '0'

    # check if all config files exist
    files_to_check = [data_config, model_config, sample_config_train, sample_config_test]
    for f in files_to_check:
        if not os.path.exists(f):
            raise Exception('File {} does not exist.'.format(f))

    # loop over alphas and identical trainings
    cmds = []
    for alpha in alphas:
        for idx in range(num_trainings):

            # make working directory
            this_outputdir = os.path.join(outputdir,
              'alpha_{:.2f}'.format(alpha).replace('.','p'), f'training_{idx}')
            os.makedirs(this_outputdir)

            # copy the config files to the working directory
            # (note: this is especially needed for the data config file,
            #  as weaver writes an auto-generated config file,
            #  which can give issues with many parallel jobs.
            #  alternatively, the --no-remake-weights option could be used,
            #  but copying the config files is probably cleaner.)
            this_data_config = os.path.join(this_outputdir, 'data_config.yaml')
            os.system(f'cp {data_config} {this_data_config}')
            this_model_config = os.path.join(this_outputdir, 'model_config.py')
            os.system(f'cp {model_config} {this_model_config}')
            this_sample_config_train = os.path.join(this_outputdir, 'sample_config_train.yaml')
            os.system(f'cp {sample_config_train} {this_sample_config_train}')
            this_sample_config_test = os.path.join(this_outputdir, 'sample_config_test.yaml')
            os.system(f'cp {sample_config_test} {this_sample_config_test}')

            # set additional network options
            # (note: the specific model must support this!)
            network_kwargs = f'flatloss_alpha {alpha}'
            if flatloss_label is not None: network_kwargs += f' flatloss_label {flatloss_label}'

            # set model prefix
            model_prefix = os.path.join(this_outputdir, 'network')

            # set output file for test results
            test_output = os.path.join(this_outputdir, 'output.root')

            # make the command
            cmd = 'weaver'
            cmd += f' --data-train {this_sample_config_train}'
            cmd += f' --data-config {this_data_config}'
            cmd += f' --network-config {this_model_config}'
            if network_kwargs is not None: cmd += f' --network-kwargs {network_kwargs}'
            cmd += f' --num-epochs {num_epochs}'
            cmd += f' --steps-per-epoch {steps_per_epoch}'
            cmd += f' --batch-size {batch_size}'
            cmd += f' --model-prefix {model_prefix}'
            cmd += f' --data-test {this_sample_config_test}'
            cmd += f' --predict-output {test_output}'
            cmd += f' --gpus {gpus}'
            # data loading options
            cmd += ' --in-memory --fetch-step 1'
            cmd += ' --copy-inputs'
            # switch between training or just printing
            if not do_training: cmd += ' --print'
            cmds.append(cmd)

    # run or submit commands
    if runmode == 'local':
        for cmd in cmds:
            #print(cmd)
            os.system(cmd)
    elif runmode=='condor':
        ct.submitCommandsAsCondorCluster('cjob_weaver', cmds,
          jobflavour='workday', conda_activate=conda_activate, conda_env=conda_env)
    elif runmode=='slurm':
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
        st.submitCommandsAsSlurmJobs(cmds, script=slurmscript, **slurm_options)
