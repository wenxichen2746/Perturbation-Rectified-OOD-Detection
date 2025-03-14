#!/bin/bash

postprocessors=(
'msp' 'pro_msp' 'neo_msp' 'pro_tempscale' 'vim' 'knn' 'odin' 'temp_scaling' 'scale' 'ash' 
'neo_msp' 'gradnorm' 'rankfeat' 'gen' 'godin' 'knn' 'vim' 'dice' 'pro_mls' 'pro_ent'
'mls'  'neo_ebo' 'neo_scale' 'neo_react' 'pro_vim' 'neo_vim'
'pro_pgddp'
)
for postprocessor in "${postprocessors[@]}"; do
    echo "Running evaluation with postprocessor: $postprocessor"

    ulimit -n 4000
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_ood.py \
        --id-data imagenet200 \
        --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
        --postprocessor "$postprocessor" \
        --batch-size 20 \
        --save-score --save-csv --savekeyword _step1_0.001
        #--overwrite
        #--savekeyword _step2_0.0005
    echo "Finished evaluation with postprocessor: $postprocessor"
done

