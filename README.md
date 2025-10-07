# Weaver framework for ML model training and evaluation on HEP data

Weaver aims at providing a streamlined yet flexible machine learning R&D framework for high energy physics (HEP) applications.
This project is forked from [hqucms/weaver-core](https://github.com/hqucms/weaver-core), see there for all details!
This specific branch is intended for training a jet classifier on Aleph simulation.

### Set up your environment

Install `miniconda` (if you don't already have it, e.g. for other projects):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Verify the installation is successful by running `conda info` and checking if the paths are pointing to your `miniconda` installation.
You might need to do `source miniconda3/bin/activate` first, if you disabled the automatic activation of the base environment.

Next, set up and activate a new conda environment:

```bash
conda create -n weaver python=3.10
conda activate weaver
```

Install pytorch. This step is a bit tricky since it depends on your OS/CUDA version.
See https://pytorch.org/get-started for more details.
The default case is simply `pip install torch`.

Then, install weaver. This will install also all the dependencies except for pytorch.
```
git clone git@github.com:LukaLambrecht/weaver-core.git
cd weaver-core
git checkout aleph
pip install -e .
```

### Configuration files

To train a neural network using `weaver`, you need to prepare:

- A YAML _data configuration file_ describing how to process the input data.
- A python _model configuration file_ providing the neural network module and the loss function.

See the [upstream repo](https://github.com/hqucms/weaver-core) for more details.
See the [configs](configs) subfolder for prepared examples for this specific case.

You also need sample lists specifying the files to be used for training and testing.
See the [weaver/configs](weaver/configs) subfolder for prepared examples for this specific case.

### Run the training and evaluation

The training and evaluation can be launched by issuing a `weaver` command with the proper command-line args.
See the [upstream repo](https://github.com/hqucms/weaver-core) for more details.
For this specific case, see [weaver/run.py](weaver/run.py) for a starting point, to be modified as the need arises.
