#!/bin/bash
FILE=$1
SKLIST=$2
MODULE_TYPE=$3

BASE=${FILE##*/}
BASE=${BASE%.*}
COMPRES=CompRes/${BASE}
DECOMPRES=DecompRes/${BASE}

python get-skmer.py --file_name $FILE --skmer_list $SKLIST

python compress.py --skmer_list $SKLIST --file_name ${BASE} --gpu 1 --output ${COMPRES} --module_type ${MODULE_TYPE}