#!/bin/bash
FILES=~/data-disk/Datasets/LJ/LJSpeech-1.1/wavs/LJ001-00[0-1]*.wav
CFG=$1
CKPT=$2
T=$3
echo "$1 $2 $3"
for f in $FILES
do
    base="$(basename -- $f)"
    out=generated/$base
    echo "$out, $f"
    python test.py $1 $2 $f $out -T $3 --amp
done