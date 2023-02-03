#!/usr/bin/env bash

python3 -m venv env

source env/bin/activate

pip install -U arbor==0.8.1 h5py==3.8.0
