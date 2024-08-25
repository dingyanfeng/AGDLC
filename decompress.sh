#!/bin/bash
FILE=$1
SKLIST=$2
MODULE_TYPE=$3
GPU=$4

BASE=${FILE##*/}
BASE=${BASE%.*}
COMPRES=CompRes/${BASE}
DECOMPRES=DecompRes/${BASE}

python decompress.py --skmer_list $SKLIST --file_name ${COMPRES} --gpu ${GPU} --output ${DECOMPRES} --module_type ${MODULE_TYPE}