#!/bin/bash
FOLDER=$1
SKLIST=$2
LOG_DIR="logs"

for FILE in $(find $FOLDER -type f); do
    BASE=${FILE##*/}
    BASE=${BASE%.*}
    COMPRES=CompRes/${BASE}
    DECOMPRES=DecompRes/${BASE}
    LOG_FILE="$LOG_DIR/${BASE}.log"

    exec > $LOG_FILE 2>&1
    echo "Processing file: $FILE"
    echo "Log file: $LOG_FILE"

    python get-skmer.py --file_name $FILE --skmer_list $SKLIST

    python compress.py --skmer_list $SKLIST --file_name ${BASE} --gpu 1 --output ${COMPRES}

    RESULT_PATH=${COMPRES}.combined
    PARAMS_PATH=${COMPRES}.params

    Fa1=$(stat --format="%s" $PARAMS_PATH)
    Fa2=$(stat --format="%s" $RESULT_PATH)
    Fb=$(stat --format="%s" $FILE)
    CR=$(echo "scale=6; ($Fa1 + $Fa2) / $Fb * 8" | bc)
    echo "Compress ratio is: $CR bits/base"

    python decompress.py --skmer_list $SKLIST --file_name ${COMPRES} --gpu 1 --output ${DECOMPRES}

    . compare.sh $DECOMPRES $FILE

    echo "Processing for $FILE complete."
done
