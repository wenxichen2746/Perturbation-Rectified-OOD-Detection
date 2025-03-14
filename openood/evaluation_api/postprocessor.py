import os
import urllib.request

from openood.postprocessors import (
    ASHPostprocessor, BasePostprocessor, ConfBranchPostprocessor,
    CutPastePostprocessor, DICEPostprocessor, DRAEMPostprocessor,
    DropoutPostProcessor, DSVDDPostprocessor, EBOPostprocessor,
    EnsemblePostprocessor, GMMPostprocessor, GodinPostprocessor,
    GradNormPostprocessor, GRAMPostprocessor, KLMatchingPostprocessor,
    KNNPostprocessor, MaxLogitPostprocessor, MCDPostprocessor,
    MDSPostprocessor, MDSEnsemblePostprocessor, MOSPostprocessor,
    ODINPostprocessor, OpenGanPostprocessor, OpenMax, PatchcorePostprocessor,
    Rd4adPostprocessor, ReactPostprocessor, ResidualPostprocessor,
    ScalePostprocessor, SSDPostprocessor, TemperatureScalingPostprocessor,
    VIMPostprocessor, RotPredPostprocessor, RankFeatPostprocessor,
    RMDSPostprocessor, SHEPostprocessor, CIDERPostprocessor, NPOSPostprocessor,
    GENPostprocessor, NNGuidePostprocessor, RelationPostprocessor,
    T2FNormPostprocessor, ReweightOODPostprocessor, fDBDPostprocessor,


    )
from openood.postprocessors.gen_react_postprocessor import GENLocalReactPostprocessor
from openood.postprocessors.pro2_ent_postprocessor import PROv2_ENT_Postprocessor
from openood.postprocessors.pro2_msp_postprocessor import PROv2_MSP_Postprocessor
from openood.postprocessors.pro2_tempscale_postprocessor import PROv2_TEMPSCALE_Postprocessor
from openood.postprocessors.pro_ebo_postprocessor import PRO_EBO_Postprocessor
from openood.postprocessors.pro_ent_postprocessor import PRO_ENT_Postprocessor
from openood.postprocessors.pro_gen_postprocessor import PRO_GENPostprocessor
from openood.postprocessors.pro_msp_postprocessor import PRO_MSP_Postprocessor
from openood.postprocessors.pro_pgddp_postprocessor import PRO_PGDDP_Postprocessor
from openood.postprocessors.pro_tempscale_postprocessor import PRO_TEMPSCALE_Postprocessor
from openood.postprocessors.ent_postprocessor import ENT_Postprocessor
from openood.utils.config import Config, merge_configs

postprocessors = {
    'fdbd': fDBDPostprocessor,
    'ash': ASHPostprocessor,
    'cider': CIDERPostprocessor,
    'conf_branch': ConfBranchPostprocessor,
    'msp': BasePostprocessor,
    'ebo': EBOPostprocessor,
    'odin': ODINPostprocessor,
    'mds': MDSPostprocessor,
    'mds_ensemble': MDSEnsemblePostprocessor,
    'npos': NPOSPostprocessor,
    'rmds': RMDSPostprocessor,
    'gmm': GMMPostprocessor,
    'patchcore': PatchcorePostprocessor,
    'openmax': OpenMax,
    'react': ReactPostprocessor,
    'vim': VIMPostprocessor,
    'gradnorm': GradNormPostprocessor,
    'godin': GodinPostprocessor,
    'mds': MDSPostprocessor,
    'gram': GRAMPostprocessor,
    'cutpaste': CutPastePostprocessor,
    'mls': MaxLogitPostprocessor,
    'residual': ResidualPostprocessor,
    'klm': KLMatchingPostprocessor,
    'temp_scaling': TemperatureScalingPostprocessor,
    'ensemble': EnsemblePostprocessor,
    'dropout': DropoutPostProcessor,
    'draem': DRAEMPostprocessor,
    'dsvdd': DSVDDPostprocessor,
    'mos': MOSPostprocessor,
    'mcd': MCDPostprocessor,
    'opengan': OpenGanPostprocessor,
    'knn': KNNPostprocessor,
    'dice': DICEPostprocessor,
    'scale': ScalePostprocessor,
    'ssd': SSDPostprocessor,
    'she': SHEPostprocessor,
    'rd4ad': Rd4adPostprocessor,
    'rotpred': RotPredPostprocessor,
    'rankfeat': RankFeatPostprocessor,
    'gen': GENPostprocessor,
    'nnguide': NNGuidePostprocessor,
    'relation': RelationPostprocessor,
    't2fnorm': T2FNormPostprocessor,
    'reweightood': ReweightOODPostprocessor,

    #PRO
    'ent': ENT_Postprocessor,
    'pro_ebo':PRO_EBO_Postprocessor,
    'pro_ent':PRO_ENT_Postprocessor,
    'pro_msp':PRO_MSP_Postprocessor,
    'pro_pgddp':PRO_PGDDP_Postprocessor,
    'pro_tempscale':PRO_TEMPSCALE_Postprocessor,
    'pro2_msp':PROv2_MSP_Postprocessor,
    'pro2_ent':PROv2_ENT_Postprocessor,
    'pro2_tempscale':PROv2_TEMPSCALE_Postprocessor,
    'pro_gen':PRO_GENPostprocessor,

    'gen_react':GENLocalReactPostprocessor,
}

link_prefix = 'https://raw.githubusercontent.com/Jingkang50/OpenOOD/main/configs/postprocessors/'


def get_postprocessor(config_root: str, postprocessor_name: str,
                      id_data_name: str):
    if 'neo' in postprocessor_name:
        postprocessor_config_path = os.path.join(config_root, 'postprocessors','neo',
                                             f'{postprocessor_name}.yml')
    else:
        postprocessor_config_path = os.path.join(config_root, 'postprocessors',
                                             f'{postprocessor_name}.yml')
    if not os.path.exists(postprocessor_config_path):
        os.makedirs(os.path.dirname(postprocessor_config_path), exist_ok=True)
        urllib.request.urlretrieve(link_prefix + f'{postprocessor_name}.yml',
                                   postprocessor_config_path)

    config = Config(postprocessor_config_path)
    config = merge_configs(config,
                           Config(**{'dataset': {
                               'name': id_data_name
                           }}))
    postprocessor = postprocessors[postprocessor_name](config)
    postprocessor.APS_mode = config.postprocessor.APS_mode
    postprocessor.hyperparam_search_done = False
    return postprocessor
