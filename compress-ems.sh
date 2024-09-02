#!/bin/bash
FILE=$1
SKLIST=$2
MODULE_TYPE=$3
GPU=$4
TIMESTEPS=$5
EMBED=$6

BASE=${FILE##*/}
BASE=${BASE%.*}
COMPRES=CompRes/${BASE}
DECOMPRES=DecompRes/${BASE}

python get-skmer.py --file_name $FILE --skmer_list $SKLIST

python compress.py --skmer_list $SKLIST --file_name ${BASE} --gpu ${GPU} --output ${COMPRES} --module_type ${MODULE_TYPE} --timesteps ${TIMESTEPS} --emb_size ${EMBED}