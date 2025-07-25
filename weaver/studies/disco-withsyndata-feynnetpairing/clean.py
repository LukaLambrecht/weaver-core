import os
import sys
import six


topdir = sys.argv[1]

toremove = []
alphadirs = os.listdir(topdir)
for alphadir in alphadirs:
    alphapath = os.path.join(topdir, alphadir)
    trainingdirs = os.listdir(alphapath)
    trainingdirs_afterremoval = trainingdirs[:]
    for trainingdir in trainingdirs:
        trainingpath = os.path.join(alphapath, trainingdir)
        resfile = os.path.join(trainingpath, 'output.root')
        if not os.path.exists(resfile):
            toremove.append(trainingpath)
            trainingdirs_afterremoval.remove(trainingdir)
    if len(trainingdirs_afterremoval)==0:
        toremove.append(alphapath)

print('Will remove the following directories:')
for d in toremove: print(f'  - {d}')

print('Continue? (y/n)')
go = six.moves.input()
if go!='y': sys.exit()

for d in toremove: os.system(f'rm -r {d}')
