# Copy samples

# Use case: privately produced samples are located on /eos on lxplus,
# but need them here to combine with background samples.


import os
import sys


if __name__=='__main__':

    # set sources
    prefix = 'llambrec@lxplus.cern.ch:'
    sources = ([
        '/eos/user/l/llambrec/hh4b-samples/ntuples-v2'
    ])

    # set target
    target = '/blue/avery/llambre1.brown/hhto4b-samples'

    # make target dir
    if not os.path.exists(target): os.makedirs(target)

    # make commands
    cmds = []
    for source in sources:
        cmd = f'scp -r {prefix}{source} {target}'
        cmds.append(cmd)

    # run commands
    for cmd in cmds:
        print(cmd)
        os.system(cmd)
