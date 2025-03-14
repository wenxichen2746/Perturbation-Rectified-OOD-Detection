#!/bin/bash


postprocessors=('gen' 'pro_gen' 'pro2_tempscale' 'pro2_msp' 'pro2_ent' 'temp_scaling'
    #'msp' 'pro_msp' 'pro2_msp'  'pro2_ent' 'pro_tempscale' 'temp_scaling'   'gradnorm' 'scale' 'ebo' 'ash' 'rankfeat' 'vim' 'react' 'odin'
#'ent' 'mls' 'pro_ebo'  'react' 'ash' 'gradnorm'  'vim' 'knn' 'scale'  'ent' 'pro_mls' 'scale' 'ash' 

)

modelids=(
'Diffenderfer2021Winning_LRR_CARD_Deck'
'Modas2021PRIMEResNet18' 'Diffenderfer2021Winning_Binary' 'Hendrycks2020AugMix_ResNeXt'
  'Diffenderfer2021Winning_LRR'  'Diffenderfer2021Winning_LRR_CARD_Deck' 'Diffenderfer2021Winning_Binary_CARD_Deck' 
)
for modelid in "${modelids[@]}"; do
    for postprocessor in "${postprocessors[@]}"; do
        echo "Running evaluation with postprocessor: $postprocessor"
        ulimit -n 4000
        CUDA_VISIBLE_DEVICES=1 python scripts/eval_ood_aarobust.py \
            --id-data cifar100 \
            --modelid "$modelid" \
            --postprocessor "$postprocessor" \
            --batch-size 20 \
            --threat corruptions \
            --save-score --save-csv \
            #--savekeyword _average_fc
            #--overwrite
        echo "Finished evaluation with postprocessor: $postprocessor"
    done
done
