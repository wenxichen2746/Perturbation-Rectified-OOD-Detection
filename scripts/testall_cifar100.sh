#!/bin/bash

# Define the list of postprocessors
postprocessors=(  'pro_pgddp'
#'msp' 'pro_msp' 'neo_msp' 'pro_tempscale' 'vim' 'knn' 'odin' 'temp_scaling' 'scale' 'ash' 
#'neo_msp' 'gradnorm' 'rankfeat' 'gen' 'godin' 'knn' 'vim' 'dice' 'pro_mls' 'pro_ent'
#'mls' 'pro_mls' 'neo_ebo' 'neo_scale' 'neo_react' 'pro_vim' 'neo_vim'
)

# Loop through each postprocessor
for postprocessor in "${postprocessors[@]}"
do
    echo "Running evaluation with postprocessor: $postprocessor"
    
    # Run the command with the current postprocessor
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_ood.py \
        --id-data cifar100 \
        --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
        --postprocessor "$postprocessor" \
        --save-score --save-csv --savekeyword _step1_0.001
    echo "Finished evaluation with postprocessor: $postprocessor"
done
