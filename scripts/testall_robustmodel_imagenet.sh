#!/bin/bash

# Define the list of postprocessors
#postprocessors=('neo_msp' 'neo_ebo' 'neo_scale' 'neo_mls' 'neo_react' 'neo_tempscale' 'neo_vim' 'ebo' 'scale' 'msp' 'ebo' 'mls' 'react' 'temp_scaling' 'vim' 'mls')
#postprocessors=('pro_msp' 'pro_pgddp' 'pro_tempscale')
postprocessors=( 'pro_gen' 'gen' 'msp' 'pro2_msp' 'ent' 'pro2_ent' 'tempscaling' 'pro2_tempscaling' 'odin' 'gradnorm'
    #'scale' 'react' 'ash' 'ebo' 'pro2_tempscale' 'pro2_msp' 'ebo' 'gradnorm' 'odin' 'msp' 'gradnorm' 'gen' 'ent'
#'react'  'knn' 'odin' 'temp_scaling'
)
#'msp' 'pro_msp'  'pro_tempscale'
modelids=(
'Erichson2022NoisyMix_new' 
 'Salman2020Do_50_2_Linf' 
'Erichson2022NoisyMix_new'  'Geirhos2018_SIN_IN'  'Hendrycks2020AugMix' 'Salman2020Do_50_2_Linf' #corruption

)
for modelid in "${modelids[@]}"; do
    for postprocessor in "${postprocessors[@]}"; do
        echo "Running evaluation with postprocessor: $postprocessor"
        ulimit -n 4000
        CUDA_VISIBLE_DEVICES=1 python scripts/eval_ood_aarobust.py \
            --id-data imagenet \
            --modelid "$modelid" \
            --postprocessor "$postprocessor" \
            --batch-size 40 \
            --save-score --save-csv  \
            --threat corruptions
            #--savekeyword _step2_0.0005 --overwrite
        echo "Finished evaluation with postprocessor: $postprocessor"
    done
done
