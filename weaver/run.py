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
    
    # data configs, generic
    # standard part without secondary vertices or V0 cands
    #data_config = os.path.abspath('configs/configs_part/standardized/data_config_parttagger_withstrange_withdedx_masked.yaml')
    # everything included
    #data_config = os.path.abspath('configs/configs_parttaggerwithv0/standardized/data_config_parttaggerwithv0.yaml')
    # everythinc included, but masked dEdx info
    #data_config = os.path.abspath('configs/configs_parttaggerwithv0/standardized/data_config_parttaggerwithv0_withdedx_masked.yaml')

    # data configs, test new input features
    #data_config = os.path.abspath('configs/configs_parttaggerwithv0/standardized/data_config_parttaggerwithv0_extrafeatures_test.yaml')
    #data_config = os.path.abspath('configs/configs_parttaggerwithv0/standardized/data_config_parttaggerwithv0_withdedx_masked_extrafeatures_test.yaml')

    # data configs, Apranik's vtag01 ntuples (different branch names)
    data_config = os.path.abspath('configs/configs_parttaggerwithv0/standardized/data_config_parttaggerwithv0_vtag01.yaml')

    # model config
    #model_config = os.path.abspath('configs/configs_part/model_config_parttagger.py')
    model_config = os.path.abspath('configs/configs_parttaggerwithv0/model_config_parttaggerwithv0.py')
    
    # sample list for training data
    #sample_config_train = os.path.abspath(f'configs/samplelists/{loc}/samples_training.yaml')
    # Apranik's vtag01 ntuples
    sample_config_train = os.path.abspath(f'configs/samplelists/{loc}/samples_training_vtag01.yaml')
    
    # sample list for testing data
    #sample_config_test = os.path.abspath(f'configs/samplelists/{loc}/samples_testing.yaml')
    # Apranik's vtag01 ntuples
    sample_config_test = os.path.abspath(f'configs/samplelists/{loc}/samples_testing_vtag01.yaml')
    
    # output dir
    #output_base = thisdir
    output_base = '/eos/user/l/llambrec/aleph-weaver-output'
    outputdir = os.path.join(output_base, 'output_test_20261012_vtag01_nepochs_30_nsteps_300_batch_512_2')
    
    # network settings
    num_epochs = 30
    steps_per_epoch = 300
    batch_size = 512
    
    # runmode
    # (choose from 'local', 'condor', and 'slurm')
    runmode = 'condor'

    # compute settings
    # GPU: use gpus = '0' to use one GPU, and gpus = None to not use a GPU.
    gpus = '0'
    # CPU: use ncpus = <requested number of CPUs>.
    # note that the number of workers is set to this number (see below).
    ncpus = 8

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

    # set log file
    # (in the current directory, next to the job log files in case of job submission;
    #  as opposed to those, this file is updated while the job is running)
    tag = os.path.basename(outputdir)
    logfile = os.path.abspath(f'cjob_weaver_{tag}_train.log')
    if os.path.exists(logfile): os.remove(logfile)

    # set output file for training and validation metrics (updated after each epoch)
    metrics_output = os.path.join(outputdir, 'metrics.json')

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
    cmd += f' --log {logfile}'
    cmd += f' --metrics-output {metrics_output}'
    
    # data loading options
    # number of CPU workers for loading the data
    cmd += f' --num-workers {ncpus}'
    # fraction of each file to read per loading step
    # (note: the memory usage is proportional to this fraction and to the number of files)
    cmd += f' --fetch-step 0.02'
    # same for the validation data
    # (can be smaller since far fewer jets are needed per epoch)
    cmd += f' --fetch-step-val 0.005'
    # load all data in memory instead of streaming
    # (not feasible for large datasets)
    #cmd += ' --in-memory --fetch-step 1'
    # copy all data to the local storage on the worker node
    # (not feasible for very large datasets)
    #cmd += ' --copy-inputs'
    
    # compute options
    if gpus is not None:
        cmd += f' --gpus {gpus}'
        # use mixed precision
        cmd += ' --use-amp'

    # run or submit commands
    if runmode == 'local':
        print(cmd)
        os.system(cmd)

    elif runmode=='condor':
        condor_options = {
            'conda_activate': f'source {miniforge}',
            'conda_env': 'weaver',
            'jobflavour': 'tomorrow',
            'cpus': ncpus,
            'mem': 64000,
            'disk': 48000
        }
        if gpus is not None and gpus != '""':
            condor_options['gpus'] = 1
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
