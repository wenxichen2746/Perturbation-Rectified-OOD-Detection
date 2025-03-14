#!/bin/bash

# Define the list of postprocessors

#'neo_msp' 'gradnorm' 'rankfeat' 'gen' 'godin' 'knn' 'vim' 'dice' 'pro_mls' 'pro_ent' 'mls' 'pro_mls' 'neo_ebo' 'neo_scale' 'neo_react' 'pro_vim' 'neo_vim'
#'pro_scale' 'pro_ebo' 'pro_ent' 'pro_mls' 'temp_scaling' 'neo_mls'
postprocessors=('gen' 'pro_gen' 'pro2_tempscale' 'pro2_msp' 'pro2_ent' 'temp_scaling')
ckpt_paths=(
    "results/imagenet_resnet50_tvsv1_base_pixmix/ckpt.pth"
    "results/imagenet_resnet50_tvsv1_augmix_default/ckpt.pth"
    "results/imagenet_resnet50_regmixup_e30_lr0.001_alpha10_default/s0/best.ckpt"
)

for postprocessor in "${postprocessors[@]}"; do
    echo "Running evaluation with postprocessor: $postprocessor"

    ulimit -n 4000
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_ood_imagenet.py \
   --tvs-pretrained \
   --arch resnet50 \
   --postprocessor "$postprocessor"  \
   --batch-size 20 \
   --save-score --save-csv 
   #--overwrite
   #--savekeyword step1_0.00001
    echo "Finished evaluation with postprocessor: $postprocessor"

    for ckpt_path in "${ckpt_paths[@]}"; do
        echo "Running evaluation with postprocessor: $postprocessor and checkpoint: $ckpt_path"
        ulimit -n 4000
        CUDA_VISIBLE_DEVICES=1 python scripts/eval_ood_imagenet.py \
            --arch resnet50 \
            --postprocessor "$postprocessor" \
            --batch-size 20 \
            --ckpt_path "$ckpt_path" \
            --save-score --save-csv
            #--overwrite
            # --savekeyword step1_0.00001
        
        echo "Finished evaluation with postprocessor: $postprocessor and checkpoint: $ckpt_path"
    done


done
