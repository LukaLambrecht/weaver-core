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
    data_config = os.path.abspath('configs/data_config_hh4b_multimh_vs_bkg_pnet.yaml')
    # model config
    model_config = os.path.abspath('configs/model_pnet.py')
    # sample list for training data
    sample_config_train = os.path.abspath('configs/samples_hh4b_multimh_vs_bkg_allyears_training.yaml')
    # sample list for testing data
    sample_config_test = os.path.abspath('configs/samples_hh4b_multimh_vs_bkg_allyears_testing.yaml')
    # output dir
    outputdir = os.path.join(thisdir, 'output_resampling_2')
    # network settings
    num_epochs = 50
    steps_per_epoch = 500
    batch_size = 256
    # specify whether to run training or only print preparatory steps
    do_training = True
    # runmode and job settings
    runmode = 'slurm'
    #conda_activate = 'source /eos/user/l/llambrec/miniforge3/bin/activate'
    #conda_env = 'weaver'
    slurmscript = 'sjob_weaver_resampling_2.sh'
    env_cmds = ([
        'source /blue/avery/llambre1.brown/miniforge3/bin/activate',
        'conda activate weaver',
        f'cd {thisdir}'
    ])

    # check if all config files exist
    files_to_check = [data_config, model_config, sample_config_train, sample_config_test]
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
    this_sample_config_train = os.path.join(outputdir, 'sample_config_train.yaml')
    os.system(f'cp {sample_config_train} {this_sample_config_train}')
    this_sample_config_test = os.path.join(outputdir, 'sample_config_test.yaml')
    os.system(f'cp {sample_config_test} {this_sample_config_test}')

    # set model prefix
    model_prefix = os.path.join(outputdir, 'network')

    # set additional network options
    # (note: the specific model must support this!)
    network_kwargs = None

    # set output file for test results
    test_output = os.path.join(outputdir, 'output.root')

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
    # data loading options
    cmd += ' --in-memory --fetch-step 1'
    cmd += ' --copy-inputs'
    # switch between training or just printing
    if not do_training: cmd += ' --print'

    # run or submit commands
    if runmode == 'local':
        print(cmd)
        os.system(cmd)
    elif runmode=='condor':
        ct.submitCommandAsCondorJob('cjob_weaver', cmd,
          jobflavour='workday', conda_activate=conda_activate, conda_env=conda_env)
    elif runmode=='slurm':
        job_name = os.path.splitext(slurmscript)[0]
        st.submitCommandAsSlurmJob(cmd, script=slurmscript,
                job_name=job_name, env_cmds=env_cmds,
                time='05:00:00')
