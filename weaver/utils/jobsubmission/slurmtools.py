########################################
# Tools for job submission using slurm #
########################################

import os
import sys


def writeSlurmScript(cmds, script, force=False,
        job_name=None, account=None, partition=None,
        output=None, error=None,
        nodes=1, ntasks_per_node=1, cpus_per_task=1,
        memory=None, time=None, constraint=None):
    """
    Write a slurm submission script.
    Input arguments:
    - cmds: a list of bash commands to execute.
    - script: the name of the script to create.
    """
    script = os.path.abspath(script)

    # handle case where requested script already exists
    if os.path.exists(script):
        if force:
            msg = f'WARNING: overwriting existing script {script}.'
            print(msg)
        else:
            msg = f'Script {script} already exists'
            msg += f' (use argument "force=True" to auto-overwrite).'
            raise Exception(msg)

    # write script
    with open(script, 'w') as f:
        # write sbatch settings
        f.write('#!/bin/bash\n')
        if job_name is not None: f.write(f'#SBATCH --job-name={job_name}\n')
        if account is not None: f.write(f'#SBATCH --account={account}\n')
        if partition is not None: f.write(f'#SBATCH --partition={partition}\n')
        if output is None: output = os.path.splitext(script)[0] + '_%j.out'
        if error is None: error = output
        f.write(f'#SBATCH --output {output}\n')
        f.write(f'#SBATCH --error {error}\n')
        f.write(f'#SBATCH --nodes={nodes}\n')
        f.write(f'#SBATCH --ntasks-per-node={ntasks_per_node}\n')
        f.write(f'#SBATCH --cpus-per-task={cpus_per_task}\n')
        if memory is not None: f.write(f'#SBATCH --mem={memory}\n')
        if time is not None: f.write(f'#SBATCH --time={time}\n')
        if constraint is not None: f.write(f'#SBATCH --constraint={constraint}\n')
        # write some convenient shell commands for bookkeeping and debugging
        f.write('echo "current host:" $(hostname)\n')
        f.write('echo "current directory:" $(pwd)\n')
        f.write('echo "current time:" $(date)\n')
        # write actual commands to run
        for cmd in cmds: f.write(cmd + '\n')
    print(f'Created slurm script {script}.')


def submitCommandAsSlurmJob(cmd, script, **kwargs):
    """Just an alias for submitCommandsAsSlurmJob for a single command"""
    submitCommandsAsSlurmJob(cmd, script, **kwargs)


def submitCommandsAsSlurmJob(cmds, script, env_cmds=None, **kwargs):
    """
    Submit a command or list of commands as a slurm job.
    Input arguments:
    - cmds: a bash command (str) or list of commands to execute.
    - script: name of the submission script to create.
    - env_cmds: list of commands to set up the environment (e.g. cmsenv or conda).
    - kwargs: passed down to writeSlurmScript
    """
    # parse commands and environment commands
    if isinstance(cmds, str): cmds = [cmds]
    if env_cmds is not None:
        if isinstance(env_cmds, str): env_cmds = [env_cmds]
        cmds = env_cmds + cmds
    # write script
    writeSlurmScript(cmds, script, **kwargs)
    # submit job
    os.system(f'sbatch {script}')


def submitCommandsAsSlurmJobs(cmds, script, env_cmds, **kwargs):
    """
    Submit a list of commands as parallel slurm jobs.
    Input arguments:
    - cmds: a list of commands to execute in parallel.
    - script: name of the submission script to create.
    - env_cmds: list of commands to set up the environment (e.g. cmsenv or conda).
    - kwargs: passed down to writeSlurmScript
    """
    # parse commands and environment commands
    if isinstance(cmds, str): cmds = [cmds]
    if env_cmds is not None:
        if isinstance(env_cmds, str): env_cmds = [env_cmds]
    # find original job name (modify in loop with index)
    job_name = None
    if 'job_name' in kwargs.keys(): job_name = kwargs.pop('job_name')
    # loop over commands
    for idx, cmd in enumerate(cmds):
        print('Preparing job {}/{}...'.format(idx+1, len(cmds)))
        # put index in script name
        thisscript = os.path.splitext(script)[0] + f'_{idx}.sh'
        thisjn = None
        if job_name is not None: thisjn = job_name + f'_{idx}'
        # merge environment commands with this command
        thiscmds = env_cmds + [cmd]
        # write script
        writeSlurmScript(thiscmds, thisscript, job_name=thisjn, **kwargs)
        # submit job
        os.system(f'sbatch {thisscript}')
