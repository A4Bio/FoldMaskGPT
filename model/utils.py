import torch
from omegaconf import OmegaConf


def load_VQAE_model(FT4Path = '/huyuqi/xmyu/FoldToken4_share/foldtoken'):
    import sys; sys.path.append(FT4Path)
    from model_interface import MInterface
    config = f'{FT4Path}/model_zoom/FT4/config.yaml'
    checkpoint = f'{FT4Path}/model_zoom/FT4/ckpt.pth'

    config = OmegaConf.load(config)
    config = OmegaConf.to_container(config, resolve=True)
    model = MInterface(**config)
    checkpoint = torch.load(checkpoint, map_location=torch.device('cuda'))
    for key in list(checkpoint.keys()):
        if '_forward_module.' in key:
            checkpoint[key.replace('_forward_module.', '')] = checkpoint[key]
            del checkpoint[key]
    model.load_state_dict(checkpoint)
    model = model.to('cuda')
    model = model.eval()
    return model

def load_VQMaskGIT_model(checkpoint):
    from model.model_interface_maskgit import MInterfaceMaskGIT
    model = MInterfaceMaskGIT.load_from_checkpoint(checkpoint).model

    model = model.to('cuda')
    model = model.eval()
    return model


def load_VQMaskGPT_model(checkpoint='/storage/huyuqi/gzy/FoldMaskGPT/model_zoom/params.ckpt', config=None):
    from data.datasets import PDBTokenizer
    from model import MInterfaceMaskGIT
    from model.models_maskgit import MaskGIT
    tokenizer = PDBTokenizer()

    config = OmegaConf.load(config)
    config = OmegaConf.to_container(config, resolve=True)
    model: MaskGIT = MInterfaceMaskGIT.load_from_checkpoint(checkpoint, **config).model
    model.to('cuda')
    return model, tokenizer