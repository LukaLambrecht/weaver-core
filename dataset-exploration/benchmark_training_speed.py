# Benchmark the training speed

# Measures separately (in jets per second):
# - the throughput of the data loader alone (no model),
# - the throughput of the model alone (training and inference steps on batches kept in memory),
#   for different batch sizes, with and without mixed precision,
# - the throughput of both together.
# This shows whether the training is limited by the data loading or by the GPU.
# Use run mode 'condor' to run on a GPU.

import os
import sys
import glob
import time
import argparse
import importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(weavercoredir)
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.nn.optimizer.ranger import Ranger
import weaver.utils.jobsubmission.condortools as ct

configdir = os.path.join(weavercoredir, 'weaver/configs/configs_parttaggerwithv0')
default_data_config = os.path.join(configdir, 'standardized/data_config_parttaggerwithv0_vtag01.yaml')
default_model_config = os.path.join(configdir, 'model_config_parttaggerwithv0.py')
default_inputdir = '/eos/experiment/fcc/ee/analyses/case-studies/aleph/processedMC/1994/zqq/stage2/VTAG-01'


def time_loader(loader, seconds, step=None):
    ### iterate over a data loader for a given time, optionally doing a training step on each batch
    start = time.time()
    first = None
    njets = 0
    for X, y, _ in loader:
        if first is None: first = time.time() - start
        if step is not None: step(X, y)
        njets += len(y['_label_'])
        if time.time() - start > seconds: break
    if step is not None and torch.cuda.is_available(): torch.cuda.synchronize()
    total = time.time() - start
    return (njets / max(total - first, 1e-9), first)


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', default=default_data_config)
    parser.add_argument('--model_config', default=default_model_config)
    parser.add_argument('--inputdir', default=default_inputdir,
      help='Directory with one subdirectory per flavour.')
    parser.add_argument('--nfiles', default=8, type=int,
      help='Number of training files to use per flavour.')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--fetch_step', default=0.01, type=float)
    parser.add_argument('--batch_sizes', default=[512, 1024, 2048], type=int, nargs='+')
    parser.add_argument('--seconds', default=120, type=float,
      help='Duration of each of the measurements involving the data loader.')
    parser.add_argument('--nsteps', default=30, type=int,
      help='Number of steps for each of the model-only measurements.')
    parser.add_argument('--runmode', default='local', choices=['local', 'condor'])
    parser.add_argument('--conda_activate', default='/eos/user/l/llambrec/miniforge3/bin/activate')
    args = parser.parse_args()

    # submit as a condor job if requested
    if args.runmode=='condor':
        cmd = 'python ' + ' '.join(sys.argv).replace('--runmode condor', '--runmode local')
        ct.submitCommandAsCondorJob('cjob_benchmark', cmd,
          conda_activate=f'source {args.conda_activate}', conda_env='weaver',
          jobflavour='microcentury', cpus=args.num_workers, gpus=1, mem=16000, disk=8000)
        sys.exit()

    # print hardware info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Host: {os.uname().nodename}, number of CPUs: {len(os.sched_getaffinity(0))}')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'GPU: {props.name} ({props.total_memory/1e9:.0f} GB)')
    else: print('GPU: none (running on CPU)')

    # find input files
    files = []
    for flavourdir in sorted(glob.glob(os.path.join(args.inputdir, 'Z*'))):
        files += sorted(glob.glob(os.path.join(flavourdir, 'QQB/train/*.root')))[:args.nfiles]
    print(f'Using {len(files)} input files.')

    # make the data loader (same settings as in train.py)
    def make_loader(batch_size):
        data = SimpleIterDataset({'': files}, args.data_config, for_training=True,
                 load_range_and_fraction=((0, 0.8), 1), fetch_step=args.fetch_step,
                 infinity_mode=True, name='benchmark')
        loader = DataLoader(data, batch_size=batch_size, drop_last=True, pin_memory=True,
                   num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
        return (loader, data.config)

    # measure the data loader alone, and keep some batches in memory
    (loader, data_config) = make_loader(max(args.batch_sizes))
    (rate, first) = time_loader(loader, args.seconds)
    print(f'[loader only] {rate:.0f} jets/s with {args.num_workers} workers (first batch after {first:.0f} s)')
    batches = []
    for X, y, _ in loader:
        batches.append((X, y))
        if len(batches) >= 4: break
    del loader

    # make the model
    spec = importlib.util.spec_from_file_location('model_config', args.model_config)
    model_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_config)
    stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    models = {use_amp: model_config.get_model(data_config, use_amp=use_amp)[0].to(device)
              for use_amp in [False, True]}
    sys.stdout = stdout
    loss_func = torch.nn.CrossEntropyLoss()
    nparams = sum(p.numel() for p in models[False].parameters())
    print(f'Number of model parameters: {nparams}')

    def make_step(use_amp):
        model = models[use_amp]
        optimizer = Ranger(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        def step(X, y):
            # (same as in train_classification in weaver/utils/nn/tools.py)
            inputs = [X[k].to(device) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = loss_func(model(*inputs), label)
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            return loss.item()
        return step

    # measure the model alone
    for use_amp in ([False, True] if torch.cuda.is_available() else [False]):
        for batch_size in args.batch_sizes:
            subbatches = [({k: v[:batch_size] for k, v in X.items()}, {k: v[:batch_size] for k, v in y.items()})
                          for X, y in batches]
            step = make_step(use_amp)
            models[use_amp].train()
            for idx in range(8): step(*subbatches[idx % len(subbatches)])
            if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.time()
            for idx in range(args.nsteps): step(*subbatches[idx % len(subbatches)])
            if torch.cuda.is_available(): torch.cuda.synchronize()
            trainrate = args.nsteps * batch_size / (time.time() - start)
            models[use_amp].eval()
            start = time.time()
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                for idx in range(args.nsteps):
                    X = subbatches[idx % len(subbatches)][0]
                    models[use_amp](*[X[k].to(device) for k in data_config.input_names])
            if torch.cuda.is_available(): torch.cuda.synchronize()
            evalrate = args.nsteps * batch_size / (time.time() - start)
            mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f'[model only] amp = {use_amp}, batch size = {batch_size}:'
                  f' training {trainrate:.0f} jets/s, inference {evalrate:.0f} jets/s, GPU memory {mem:.1f} GB', flush=True)

    # measure the data loader and the model together
    for use_amp in ([False, True] if torch.cuda.is_available() else [False]):
        for batch_size in [min(args.batch_sizes), max(args.batch_sizes)]:
            (loader, _) = make_loader(batch_size)
            models[use_amp].train()
            (rate, first) = time_loader(loader, args.seconds, step=make_step(use_amp))
            print(f'[loader + model] amp = {use_amp}, batch size = {batch_size}: training {rate:.0f} jets/s', flush=True)
            del loader

    # hard exit to avoid waiting for the data loader workers
    sys.stdout.flush()
    os._exit(0)
