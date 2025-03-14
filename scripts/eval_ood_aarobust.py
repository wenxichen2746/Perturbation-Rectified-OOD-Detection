import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
#print(ROOT_DIR)
sys.path.append(ROOT_DIR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import argparse
import pickle
import collections
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

from openood.evaluation_api import Evaluator

from robustbench.utils import load_model
#from torchvision import transforms as trn
#from openood.evaluation_api.preprocessor import Convert
from openood.evaluation_api.preprocessor import *
from openood.utils import Config
from torchvision import transforms as trn

parser = argparse.ArgumentParser()
#parser.add_argument('--root', required=True)
parser.add_argument('--postprocessor', default='msp')
parser.add_argument(
    '--id-data',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100', 'aircraft', 'cub', 'imagenet200','imagenet'])
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--save-csv', action='store_true')
parser.add_argument('--save-score', action='store_true')
parser.add_argument('--fsood', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--savekeyword', default='', type=str)
parser.add_argument('--modelid', default='Wang2023Better_WRN-28-10', type=str)#for robustbench model
parser.add_argument('--threat',default='corruptions',type=str)
args = parser.parse_args()


root = os.path.join(ROOT_DIR, 'results','robustmodels',args.id_data,args.threat,args.modelid)
if not os.path.exists(root):
    os.makedirs(root)
print(root)
# specify an implemented postprocessor
# 'openmax', 'msp', 'temp_scaling', 'odin'...
postprocessor_name = args.postprocessor


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# load pre-setup postprocessor if exists
if os.path.isfile(
        os.path.join(root, 'postprocessors',
                    f'{postprocessor_name}.pkl')) and not args.overwrite:
    with open(
            os.path.join(root, 'postprocessors',
                        f'{postprocessor_name}.pkl'), 'rb') as f:
        postprocessor = pickle.load(f)
else:
    postprocessor = None

# load the pretrained model provided by the user


model = load_model(model_name=args.modelid,dataset=args.id_data, threat_model=args.threat)#Linf
device = torch.device("cuda:0") 
model=model.to(device)

config = Config(**default_preprocessing_dict[args.id_data])
if args.id_data=='cifar10' or args.id_data=='cifar100':
    preprocessor = TestRobustmodelPreProcessor(config)
elif args.id_data=='imagenet':
    preprocessor = trn.Compose([
    trn.Resize(256),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
else:
    raise NotImplementedError

evaluator = Evaluator(
    model,
    id_name=args.id_data,  # the target ID dataset
    data_root=os.path.join(ROOT_DIR, 'data'),
    config_root=os.path.join(ROOT_DIR, 'configs'),
    preprocessor=preprocessor,  
    postprocessor_name=postprocessor_name,
    postprocessor=postprocessor,  # the user can pass his own postprocessor as well
    batch_size=args.batch_size,  # for certain methods the results can be slightly affected by batch size
    shuffle=False,
    num_workers=8,
    robustbench=True)

# load pre-computed scores if exist
if not args.overwrite:
    if os.path.isfile(
            os.path.join(root, 'scores', f'{postprocessor_name}.pkl')):
        with open(
                os.path.join(root, 'scores', f'{postprocessor_name}.pkl'),
                'rb') as f:
            scores = pickle.load(f)
        update(evaluator.scores, scores)
        print('Loaded pre-computed scores from file.')

# save the postprocessor for future reuse
if hasattr(evaluator.postprocessor, 'setup_flag'
            ) or evaluator.postprocessor.hyperparam_search_done is True:
    pp_save_root = os.path.join(root, 'postprocessors')
    if not os.path.exists(pp_save_root):
        os.makedirs(pp_save_root)

    pp_save_filename = f'{postprocessor_name}_{args.savekeyword}.pkl' if args.savekeyword else f'{postprocessor_name}.pkl'
    if not os.path.isfile(os.path.join(pp_save_root, pp_save_filename)) or args.overwrite:
        with open(os.path.join(pp_save_root, pp_save_filename), 'wb') as f:
            pickle.dump(evaluator.postprocessor, f, pickle.HIGHEST_PROTOCOL)

metrics = evaluator.eval_ood(fsood=args.fsood)


# save computed scores
score_save_filename = f'{postprocessor_name}_{args.savekeyword}.pkl' if args.savekeyword else f'{postprocessor_name}.pkl'
if args.save_score:
    score_save_root = os.path.join(root, 'scores')
    if not os.path.exists(score_save_root):
        os.makedirs(score_save_root)
    with open(os.path.join(score_save_root, score_save_filename), 'wb') as f:
        pickle.dump(evaluator.scores, f, pickle.HIGHEST_PROTOCOL)

# saving and recording
if args.save_csv:
    csv_filename = f'{postprocessor_name}_{args.savekeyword}.csv' if args.savekeyword else f'{postprocessor_name}.csv'
    saving_root = os.path.join(root, 'ood' if not args.fsood else 'fsood')
    if not os.path.exists(saving_root):
        os.makedirs(saving_root)
    metrics.to_csv(os.path.join(saving_root,csv_filename),float_format='{:.2f}'.format)

