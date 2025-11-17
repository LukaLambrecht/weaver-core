import os
import sys
import six
import glob

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(weavercoredir)
import weaver.utils.jobsubmission.condortools as ct
import weaver.utils.jobsubmission.slurmtools as st


if __name__=='__main__':

    # settings
    input_file_pattern = '/eos/user/l/llambrec/aleph-data/ntuples_jetlevel/mc/output_qqb_*_train.root'
    upsample_factor = 3
    repeats = 1
    std = 0.1
    outputdir = '/eos/user/l/llambrec/aleph-data/ntuples_jetlevel_upsampled'
    runmode = 'condor'

    # make output directory
    if not os.path.exists(outputdir): os.makedirs(outputdir)

    # find input files
    inputfiles = sorted(glob.glob(input_file_pattern))
    print(f'Found {len(inputfiles)} input files:')
    for f in inputfiles: print(f'  - {f}')
    print('Continue? (y/n)')
    go = six.moves.input()
    if go != 'y': sys.exit()

    # loop over input files and number of repeats
    cmds = []
    for inputfile in inputfiles:
        for repeatidx in range(repeats):
            
            # make output file
            outputfile = os.path.basename(inputfile).replace('.root', f'_batch{repeatidx}.root')
            outputfile = os.path.join(outputdir, outputfile)

            # make the command
            cmd = 'python naive_upsampling.py'
            cmd += f' -i {inputfile}'
            cmd += f' -o {outputfile}'
            cmd += f' -f {upsample_factor}'
            cmd += f' -s {std}'
            
            # add to list
            cmds.append(cmd)

    # run or submit commands
    if runmode == 'local':
        for cmd in cmds:
            print(cmd)
            os.system(cmd)
    elif runmode=='condor':
        conda_activate = 'source /eos/user/l/llambrec/miniforge3/bin/activate'
        conda_env = 'weaver'
        ct.submitCommandsAsCondorCluster('cjob_upsample', cmds,
          jobflavour='workday', conda_activate=conda_activate, conda_env=conda_env)
    elif runmode=='slurm':
        slurmscript = 'sjob_upsample.sh'
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
        st.submitCommandsAsSlurmJobs(cmds, slurmscript, **slurm_options)
