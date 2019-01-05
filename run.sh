#!/bin/bash

DATADIR=data/sine_data
CONFIGFILE=configs/sine_config/sine_2dim_without_reduction.json
RESULTDIR=results

python experiment.py ${DATADIR} ${CONFIGFILE} ${RESULTDIR}

