#!/bin/bash

# Define the list of postprocessors
postprocessors=('pro_msp' 'pro_pgddp' 'pro_tempscale'
'msp'  'ebo' 'mls' 'pro_msp' 'knn' 'odin' 'scale' 'temp_scaling'  'pro_tempscale'  'pro_mls'
  'ent' 'mls' 'pro_ebo'  'react' 'ash' 'gradnorm'  'vim' 'knn' 'scale' 'pro_ent' 'ent' 'pro_mls' 'vim' 'scale' 'ash' 
'pro_tempscale' 'gradnorm' 'scale' 'ebo' 'ash' 'rankfeat' 'vim' 'react' 'odin'
)

modelids=(
 
'Diffenderfer2021Winning_Binary' 'Kireev2021Effectiveness_RLATAugMix' 'Hendrycks2020AugMix_ResNeXt' 'Diffenderfer2021Winning_LRR_CARD_Deck' 'Diffenderfer2021Winning_Binary_CARD_Deck' 
'Diffenderfer2021Winning_LRR' 'Diffenderfer2021Winning_LRR_CARD_Deck' 'Kireev2021Effectiveness_RLATAugMix' 'Hendrycks2020AugMix_ResNeXt'

)
for modelid in "${modelids[@]}"; do
    for postprocessor in "${postprocessors[@]}"; do
        echo "Running evaluation with postprocessor: $postprocessor"
        ulimit -n 4000
        CUDA_VISIBLE_DEVICES=0 python scripts/eval_ood_aarobust.py \
            --id-data cifar10 \
            --modelid "$modelid" \
            --postprocessor "$postprocessor" \
            --batch-size 20 \
            --threat corruptions \
            --save-score --save-csv\
            --savekeyword _step1_0.0005
            #--overwrite
        echo "Finished evaluation with postprocessor: $postprocessor"
    done
done
