import os
import sys
import json
import shutil
import numpy as np  

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(weavercoredir)
import weaver.utils.jobsubmission.condortools as ct
import weaver.utils.jobsubmission.slurmtools as st


if __name__=='__main__':

    # settings based on user
    user = os.getenv('USER')
    miniforge = None
    loc = 'oscar'
    if user=='llambre1.brown':
        miniforge = '/blue/avery/llambre1.brown/miniforge3/bin/activate'
        loc = 'uflhpg'
    if user=='llambrec':
        miniforge = '/eos/user/l/llambrec/miniforge3/bin/activate'
        loc = 'lxplus'
    if user=='tgillin':
        miniforge = '/users/tgillin/miniconda3/etc/profile.d/conda.sh'
        loc = 'oscar'

    # common settings
    weaverdir = os.path.join(weavercoredir, 'weaver')
    
    # data config
    #data_config = os.path.abspath('configs/configs_part/standardized/data_config_parttagger_withstrange_withdedx_masked.yaml')
    #data_config = os.path.abspath('configs/configs_parttaggerwithv0/standardized/data_config_parttaggerwithv0_withdedx_masked.yaml') # masked dEdx
    data_config = os.path.abspath('configs/configs_parttaggerwithv0/standardized/data_config_parttaggerwithv0.yaml') # everything included
    
    # model config
    #model_config = os.path.abspath('configs/configs_part/model_config_parttagger.py')
    model_config = os.path.abspath('configs/configs_parttaggerwithv0/model_config_parttaggerwithv0.py')
    
    # sample list for training data
    sample_config_train = os.path.abspath(f'configs/samplelists/{loc}/samples_training.yaml')
    
    # sample list for testing data
    sample_config_test = os.path.abspath(f'configs/samplelists/{loc}/samples_testing.yaml')
    
    # output dir
    #output_base = thisdir
    output_base = '/eos/user/l/llambrec/aleph-weaver-output'
    outputdir = os.path.join(output_base, 'output_test_nepochs_180_nsteps_300')
    
    # network settings
    num_epochs = 180
    steps_per_epoch = 300
    batch_size = 512
    
    # runmode and job settings
    # (choose from 'local', 'condor', and 'slurm')
    runmode = 'condor'
    gpus= '0'
    #gpus = None

    # check if all config files exist
    files_to_check = [data_config, model_config, sample_config_train, sample_config_test, miniforge]
    for f in files_to_check:
        if not os.path.exists(f):
            raise Exception('File {} does not exist.'.format(f))

    # make output directory (remove if it already exists)
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
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
    cmd = 'python train.py'
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
        condor_options = {
            'conda_activate': f'source {miniforge}',
            'conda_env': 'weaver',
            'jobflavour': 'tomorrow',
            'cpus': 4,
            'mem': 16000,
            'disk': 32000
        }
        if gpus is not None and gpus != '""':
            condor_options['gpus'] = 1
        tag = os.path.basename(outputdir)
        ct.submitCommandAsCondorJob(f'cjob_weaver_{tag}', cmd, **condor_options)
    
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
        if gpus is not None and gpus != '""':
            #slurm_options['partition'] = 'gpu'
            #slurm_options['gres'] = 'gpu:1'
            slurm_options['gpus'] = '1'
        st.submitCommandAsSlurmJob(cmd, slurmscript, **slurm_options)
