# sh ./scripts/sweep/sweep_posthoc-backup.sh
python ./scripts/sweep/sweep_posthoc.py \
--benchmarks 'cifar10' 'cifar100' \
--methods 'neo_msp' 'neo_ebo' 'neo_scale' 'neo_mls' 'neo_react' 'neo_tempscale' 'neo_vim' 'ebo' 'scale' 'msp' 'ebo' 'mls' 'react' 'temp_scaling' 'vim' 'mls' \
--metrics 'ood' \
--metric2save 'fpr95' 'auroc' 'aupr_in' \
--output-dir './results/ood' \
--launcher 'local' \
--update_form_only
