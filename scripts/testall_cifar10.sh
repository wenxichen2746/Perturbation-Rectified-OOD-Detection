#!/bin/bash

# Define the list of postprocessors to be tested here
postprocessors=('pro_gen' 'gradnorm' 'odin' 'dice' 'scale' 'pro2_msp'
)
for postprocessor in "${postprocessors[@]}"
do
    echo "Running evaluation with postprocessor: $postprocessor"
    ulimit -n 4000
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_ood.py \
        --id-data cifar10 \
        --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
        --postprocessor "$postprocessor" \
        --save-score --save-csv 
        #--savekeyword _step2_0.0005
    echo "Finished evaluation with postprocessor: $postprocessor"
done

