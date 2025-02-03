import os
import sys
import argparse

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(weavercoredir)
import weaver.utils.jobsubmission.slurmtools as slurmtools


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--runmode', default='local', choices=['local', 'slurm'])
    args = parser.parse_args()

    # define local running
    if args.runmode=='local': print('Hello world!')

    # define running via slurm
    elif args.runmode=='slurm':

        # define script name
        script = 'sjob_test.sh'
        job_name = os.path.splitext(script)[0]

        # define commands to set the environment
        thisdir = os.path.abspath(os.path.dirname(__file__))
        env_cmds = ([
          'source /blue/avery/llambre1.brown/miniforge3/bin/activate',
          'conda activate weaver',
          f'cd {thisdir}'
        ])
    
        # define commands to execute
        cmds = ['python3 test_slurm_job.py --runmode local']

        # submit the job
        slurmtools.submitCommandsAsSlurmJob(cmds, script, env_cmds=env_cmds,
            job_name=job_name, account=None, partition=None,
            output=None, error=None,
            nodes=1, ntasks_per_node=1, cpus_per_task=1,
            memory=None, time=None)
