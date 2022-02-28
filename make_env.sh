#!/usr/bin/env bash

set -x
set -e

mkdir ${PWD}/tmp
export TMPDIR=${PWD}/tmp #in case the default tmp dir is not big enough to install pytorch

python3 -m venv env
source env/bin/activate

pip install --upgrade pip

#pip install torch
#pip install numpy
#pip install matplotlib
#pip install pandas
#pip install sklearn
#pip install tqdm
#pip install uproot
#pip install xgboost==1.2.1 #latest release not to throw error due to negative weights

pip install -r requirements.txt

rm -r ${PWD}/tmp
