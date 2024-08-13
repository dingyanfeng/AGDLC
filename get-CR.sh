FILE=$1
BASE=${FILE##*/}
BASE=${BASE%.*}
COMPRES=CompRes/${BASE}
RESULT_PATH=${COMPRES}.combined
PARAMS_PATH=${COMPRES}.params

Fa1=$(stat --format="%s" $PARAMS_PATH)
Fa2=$(stat --format="%s" $RESULT_PATH)
Fb=$(stat --format="%s" $FILE)
CR=$(echo "scale=6; ($Fa1 + $Fa2) / $Fb * 8" | bc)
echo "Compress ratio is: $CR bits/base"