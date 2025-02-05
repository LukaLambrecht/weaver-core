import os
import sys

thisdir = os.path.abspath(os.path.dirname(__file__))
import weaver.utils.jobsubmission.slurmtools as st


if __name__=='__main__':

    # read the weaver command from the command line
    args = sys.argv[1:]

    # conda environment settings
    env_cmds = ([
        'source /blue/avery/llambre1.brown/miniforge3/bin/activate',
        'conda activate weaver',
        f'cd {thisdir}'
    ])

    # make the full weaver command
    cmd = 'weaver ' + ' '.join(args)
    print('Submitting the following command:')
    print(cmd)

    # submit the job
    script = 'sjob_weaver.sh'
    job_name = os.path.splitext(script)[0]
    st.submitCommandsAsSlurmJob([cmd], script=script,
        job_name=job_name, env_cmds=env_cmds)
