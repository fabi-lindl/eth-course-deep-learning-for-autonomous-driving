from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR

from mtl.datasets.dataset_miniscapes import DatasetMiniscapes
from mtl.models.model_deeplab_v3_plus import ModelDeepLabV3Plus
from mtl.models.model_deeplab_v3_plus_branched import ModelDeepLabV3PlusBranched
from mtl.models.model_deeplab_v3_plus_sa import ModelDeepLabV3PlusBranchedSA
from mtl.models.model_deeplab_v3_plus_sa_deeper import ModelDeepLabV3PlusBranchedSADeeper

def resolve_dataset_class(name):
    return {
        'miniscapes': DatasetMiniscapes,
    }[name]

def resolve_model_class(name):
    return {
        # Task 1. 
        'deeplabv3p': ModelDeepLabV3Plus,
        # Task 2. 
        'deeplabv3p_branched': ModelDeepLabV3PlusBranched,
        # Task 2. 
        'deeplabv3p_sa': ModelDeepLabV3PlusBranchedSA,
        'deeplabv3p_sa_deeper': ModelDeepLabV3PlusBranchedSADeeper,
    }[name]

def resolve_optimizer(cfg, params):
    if cfg.optimizer == 'sgd':
        return SGD(
            params,
            lr=cfg.optimizer_lr,
            momentum=cfg.optimizer_momentum,
            weight_decay=cfg.optimizer_weight_decay,
        )
    elif cfg.optimizer == 'adam':
        return Adam(
            params,
            lr=cfg.optimizer_lr,
            weight_decay=cfg.optimizer_weight_decay,
        )
    else:
        raise NotImplementedError

def resolve_lr_scheduler(cfg, optimizer):
    if cfg.lr_scheduler == 'poly':
        return LambdaLR(
            optimizer,
            lambda ep: max(1e-6, (1 - ep / cfg.num_epochs) ** cfg.lr_scheduler_power)
        )
    else:
        raise NotImplementedError
