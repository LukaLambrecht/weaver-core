import os
import sys
import json
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(weavercoredir)
import weaver.utils.condortools as ct


if __name__=='__main__':

    # common settings
    weaverdir = os.path.join(weavercoredir, 'weaver')
    # data config
    data_config = os.path.join(weaverdir, 'configs-data/hh4b_simplenn_disco.yaml')
    # model config
    model_config = os.path.join(weaverdir, 'configs-model/simple_nn_withdisco.py')
    # base output dir
    base_outputdir = os.path.join(thisdir, 'output_test')
    # training and testing data
    data_train_pattern = '/eos/user/l/llambrec/HHto4b/ntuples-v2/ntuples_*_0L/mc/pieces/HHto4b_mH_100_powheg_pythia8_Run3_*'
    data_train_pattern += ' /eos/user/l/llambrec/HHto4b/ntuples-v2/ntuples_*_0L/mc/pieces/HHto4b_mH_150_powheg_pythia8_Run3_*'
    data_test_pattern = '/eos/user/l/llambrec/HHto4b/ntuples-v2/ntuples_2022_0L/mc/pieces/HHto4b_mH_100_powheg_pythia8_Run3_*'
    data_test_pattern += ' /eos/user/l/llambrec/HHto4b/ntuples-v2/ntuples_2022_0L/mc/pieces/HHto4b_mH_150_powheg_pythia8_Run3_*'
    # alpha range
    alphas = np.concatenate((np.linspace(0, 1, num=11), np.linspace(1.5, 3, num=4)))
    # network settings
    architecture = [16, 8, 4]
    num_epochs = 30
    steps_per_epoch = 300
    batch_size = 128
    num_trainings = 5
    # runmode and job settings
    runmode = 'condor'
    conda_activate = 'source /eos/user/l/llambrec/miniforge3/bin/activate'
    conda_env = 'weaver'

    # loop over alphas and identical trainings
    cmds = []
    for alpha in alphas:
        for idx in range(num_trainings):

            # make working directory
            outputdir = os.path.join(base_outputdir,
              'alpha_{:.2f}'.format(alpha).replace('.','p'), f'training_{idx}')
            os.makedirs(outputdir)

            # set model prefix
            model_prefix = os.path.join(outputdir, 'network')

            # set additional network options
            # (note: the specific model must support this!)
            arch_str = json.dumps(architecture).replace(' ','')
            network_kwargs = f'disco_alpha {alpha}'
            network_kwargs += f' disco_power 1'
            network_kwargs += f' architecture {arch_str}'

            # set output file for test results
            test_output = os.path.join(outputdir, 'output.root')

            # make the command
            cmd = 'weaver'
            cmd += f' --data-train {data_train_pattern}'
            cmd += f' --data-config {data_config}'
            cmd += f' --network-config {model_config}'
            cmd += f' --network-kwargs {network_kwargs}'
            cmd += ' --in-memory --fetch-step 1'
            cmd += f' --num-epochs {num_epochs}'
            cmd += f' --steps-per-epoch {steps_per_epoch}'
            cmd += f' --batch-size {batch_size}'
            cmd += f' --model-prefix {model_prefix}'
            cmd += f' --data-test {data_test_pattern}'
            cmd += f' --predict-output {test_output}'
            cmds.append(cmd)

    # run or submit commands
    if runmode == 'local':
        for cmd in cmds: os.system(cmd)
    elif runmode=='condor':
        ct.submitCommandsAsCondorCluster('cjob_weaver', cmds,
          jobflavour='workday', conda_activate=conda_activate, conda_env=conda_env)
