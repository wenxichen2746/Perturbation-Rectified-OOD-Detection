#!/bin/bash
# sh scripts/ood/godin/imagenet200_test_ood_godin.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

# ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_godin_net_godin_e90_lr0.1_default \
   --postprocessor godin \
   --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_godin_net_godin_e90_lr0.1_default \
   --postprocessor godin \
   --save-score --save-csv --fsood
