#!/bin/bash

# Define the list of postprocessors
postprocessors=( 'pro_pgddp'
'pro_ebo' 'pro_ent' 'pro_vim'  'scale' 'msp' 'ebo' 'mls' 'react'  'vim' 
)


for postprocessor in "${postprocessors[@]}"
do
    echo "Running evaluation with postprocessor: $postprocessor"

    ulimit -n 4000
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor "$postprocessor"  \
    --batch-size 20 \
    --save-score --save-csv --savekeyword _step1_0.001 --overwrite
    echo "Finished evaluation with postprocessor: $postprocessor"
done
