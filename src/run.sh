#!/bin/bash
# makes and runs both haze removal implementations

SEQ_BIN=haze_removal_seq
CUDA_BIN=haze_removal_cuda

INPUT_DIR=../images
SEQ_OUTPUT_DIR=../seq_results
CUDA_OUTPUT_DIR=../cuda_results

TIME_STRING="Computation Time"
TIME_UNIT="seconds"

function haze_removal {
    BIN="$1"
    OUTPUT_DIR="$2"
    TOTAL_TIME=0
    # sum computation time for every picture in input
    for INFILE in "$INPUT_DIR"/*.png;
    do
        echo -n "Running haze removal on $INFILE..."
        # use the same filename for output
        OUTFILE="$OUTPUT_DIR"${INFILE#"$INPUT_DIR"}
        TIME=$(./"$BIN" -i "$INFILE" -o "$OUTFILE" | grep "$TIME_STRING" | awk '{printf $3}')
        TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME" | bc)
        ERROR=$?
        if [[ $ERROR == 0 ]]
        then
            echo "done"
        else
            echo "Error"
            exit
        fi
    done

    echo "$TOTAL_TIME" $TIME_UNIT | awk '{printf "\nTotal Time: %.3f %s\n\n", $1, $2}'
}

if [[ "$1" == "help" ]]
then
    echo "Usage: ./$0 [help] [seq] [cuda]"
    echo ""
    echo "help:    shows this message"
    echo "seq:     runs sequential haze removal"
    echo "cuda:    runs cuda haze removal"
    echo "default: runs seq and cuda and computes speedup"
    exit
fi

# only show errors
make clean > /dev/null
make > /dev/null

if [[ "$1" == "seq" ]]
then
    echo "Sequential Haze Removal"
    haze_removal "$SEQ_BIN" "$SEQ_OUTPUT_DIR"
    exit
fi

if [[ "$1" == "cuda" ]]
then
    echo "CUDA Haze Removal"
    haze_removal "$CUDA_BIN" "$CUDA_OUTPUT_DIR"
    exit
fi

if [[ -f "$SEQ_BIN" ]]
then
    echo "Sequential Haze Removal"
    haze_removal "$SEQ_BIN" "$SEQ_OUTPUT_DIR"
    TOTAL_SEQ_TIME=$TOTAL_TIME
else
    echo "$SEQ_BIN not found"
fi

if [[ -f "$CUDA_BIN" ]]
then
    echo "CUDA Haze Removal"
    haze_removal "$CUDA_BIN" "$CUDA_OUTPUT_DIR"
    TOTAL_CUDA_TIME=$TOTAL_TIME
else
    echo "$CUDA_BIN not found"
fi

# compute speedup relative to seq
if [[ -f "$SEQ_BIN" ]] && [[ -f "$CUDA_BIN" ]]
then
    echo "$TOTAL_SEQ_TIME" "$TOTAL_CUDA_TIME" | awk '{printf "\nSpeedup: %.1f%%\n\n", (100 * ($1 - $2) / $1)}'
fi
