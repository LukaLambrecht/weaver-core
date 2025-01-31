import os
import sys

import weaver.utils.condortools as ct


if __name__=='__main__':

    # read the weaver command from the command line
    args = sys.argv[1:]

    # conda environment settings
    conda_activate = 'source /eos/user/l/llambrec/miniforge3/bin/activate'
    conda_env = 'weaver'

    # make the full weaver command
    cmd = 'weaver ' + ' '.join(args)
    print('Submitting the following command:')
    print(cmd)

    # submit the job
    ct.submitCommandAsCondorJob('cjob_weaver', cmd,
      jobflavour='workday', conda_activate=conda_activate, conda_env=conda_env)
