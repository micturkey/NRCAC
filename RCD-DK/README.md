## Introduction
RCD-DK algorithm is based on [RCD algorithm](https://github.com/azamikram/rcd/tree/master) that is source code for [Root Cause Analysis of Failures in Microservices through Causal Discovery](https://proceedings.neurips.cc/paper_files/paper/2022/file/c9fcd02e6445c7dfbad6986abee53d0d-Paper-Conference.pdf).

Below are the steps to run the RCD-DK algorithm, modifed from the original RCD algorithm's instructions.

## Setup
The following insutrctions assume that you are running Ubuntu-20.04.
#### Install python env
```bash
sudo apt update
sudo apt install -y build-essential \
                    python-dev \
                    python3-venv \
                    python3-pip \
                    libxml2 \
                    libxml2-dev \
                    zlib1g-dev \
                    python3-tk \
                    graphviz

cd ~
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
```

#### Install dependencies
```bash
git clone https://github.com/micturkey/NRCAC.git
cd NRCAC/RCD-DK
pip install -r requirements.txt
```

#### Link modifed files
To implement RCD-DK, we modified some code from pyAgrum and causal-learn.
Some of these changes expose some internal information for reporting results (for example number of CI tests while executing PC) or modify the existing behaviour (`local_skeleton_discovery` in `SekeletonDiscovery.py` implements the localized approach for RCD and **domain knowledge-based approach for RCD-DK**). A few of these changes also fix some minor bugs.

Assuming the rcd repository was cloned at home, execute the following;
```bash
ln -fs ~/NRCAC/RCD-DK/pyAgrum/lib/image.py ~/env/lib/python3.8/site-packages/pyAgrum/lib/
ln -fs ~/NRCAC/RCD-DK/causallearn/search/ConstraintBased/FCI.py ~/env/lib/python3.8/site-packages/causallearn/search/ConstraintBased/
ln -fs ~/NRCAC/RCD-DK/causallearn/utils/Fas.py ~/env/lib/python3.8/site-packages/causallearn/utils/
ln -fs ~/NRCAC/RCD-DK/causallearn/utils/PCUtils/SkeletonDiscovery.py ~/env/lib/python3.8/site-packages/causallearn/utils/PCUtils/
ln -fs ~/NRCAC/RCD-DK/causallearn/graph/GraphClass.py ~/env/lib/python3.8/site-packages/causallearn/graph/
```

## Using RCD

#### Generate Synthetic Data
```sh
./gen_data.py
```

#### Executing RCD-DK with Synthetic Data
```sh
./rcd.py --path [PATH_TO_DATA] --local --k 3 --knowledge=True
```

`--local` options enables the localized RCD while `--k` estimates the top-`k` root causes.  
`--knowledge=True` option enables the knowledge-based approach for RCD-DK.

## Dataset
In the `data` directory, we provide the synthetic dataset used in the paper.  
In the `dataset` directory, we provide the train-ticket dataset used in the paper.  
All the datasets are examples and are used for demonstration and test purposes only.  
Sock-shop datasets can be downloaded from [BARO](https://github.com/phamquiluan/baro?tab=readme-ov-file#download-datasets).

`collectdata.py` and `builddep2.py` in each dataset directory are eCollection's scripts.