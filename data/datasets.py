import json
import random

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .pdb_tokenizer import PDBTokenizer


class MultiFoldLang(nn.Module):
    def __init__(self, map_path=f'/huyuqi/xmyu/FoldGPT/FoldToken4/model_zoom'):
        super().__init__()
        self.register_buffer('map12to5', torch.load(f'{map_path}/map12to5.pt').cpu())
        self.register_buffer('map12to6', torch.load(f'{map_path}/map12to6.pt').cpu())
        self.register_buffer('map12to7', torch.load(f'{map_path}/map12to7.pt').cpu())
        self.register_buffer('map12to8', torch.load(f'{map_path}/map12to8.pt').cpu())
        self.register_buffer('map12to9', torch.load(f'{map_path}/map12to9.pt').cpu())
        self.register_buffer('map12to10', torch.load(f'{map_path}/map12to10.pt').cpu())
        self.register_buffer('map12to11', torch.load(f'{map_path}/map12to11.pt').cpu())

        self.register_buffer('map5to12', torch.load(f'{map_path}/map5to12.pt').cpu())
        self.register_buffer('map6to12', torch.load(f'{map_path}/map6to12.pt').cpu())
        self.register_buffer('map7to12', torch.load(f'{map_path}/map7to12.pt').cpu())
        self.register_buffer('map8to12', torch.load(f'{map_path}/map8to12.pt').cpu())
        self.register_buffer('map9to12', torch.load(f'{map_path}/map9to12.pt').cpu())
        self.register_buffer('map10to12', torch.load(f'{map_path}/map10to12.pt').cpu())
        self.register_buffer('map11to12', torch.load(f'{map_path}/map11to12.pt').cpu())


def schedule_maskgit_ratio(unknown_number, ratio=None, method="cosine"):
    """🔍 Adapted from https://github.com/google-research/maskgit/blob/main/maskgit/libml/mask_schedule.py
    Generates a mask by scheduling mask functions R.

    Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. During
    training, the input ratio is uniformly sampled; during inference, the input
    ratio is based on the step number divided by the total iteration number: t/T.
    Based on experiements, we find that masking more in training helps.
    Args:
      unknown_number: The total number of tokens that can be masked out.
      ratio: The uniformly sampled ratio [0, 1) as input.
      method: implemented functions are ["uniform", "cosine", "pow", "log", "exp"]
        "pow2.5" represents x^2.5

    Returns:
      The mask (BoolTensor).
    """
    if ratio is None:  # training
        ratio = random.uniform(0, 1)

    if method == "uniform":
        mask_ratio = 1. - ratio
    elif "pow" in method:
        exponent = float(method.replace("pow", ""))
        mask_ratio = 1. - ratio ** exponent
    elif method == "cosine":
        mask_ratio = math.cos(math.pi / 2. * ratio)
    elif method == "log":
        mask_ratio = -math.log2(ratio) / math.log2(unknown_number)
    elif method == "exp":
        mask_ratio = 1 - math.exp2(-math.log2(unknown_number) * (1 - ratio))
    else:
        raise ValueError(f"Unrecognized masking method: {method}")

    mask_ratio = max(1e-6, min(mask_ratio, 1.0))  # Clamps mask into [epsilon, 1)
    return mask_ratio


def generate_mask(length, mode='scaffolding', **kwargs):
    if mode in ('scaffolding', 'inpainting'):
        # Motif的长度在5到30之间
        motif_length = random.randint(5, int(length * 0.5))
        # 随机选择motif的起始位置
        motif_start = random.randint(0, length - motif_length)  # 第一个位置是BOS token，不掩码
        # 获取motif的结束位置
        motif_end = motif_start + motif_length

    if mode == 'scaffolding':
        # scaffolding模式，掩码掉motif以外的所有位置
        mask = [i for i in range(length) if i < motif_start or i >= motif_end]
    elif mode == 'inpainting':
        # inpainting模式，掩码掉motif
        mask = [i for i in range(motif_start, motif_end)]
    elif mode == 'MLM':
        mask = [i for i in range(length) if random.random() < 0.15]
    elif mode == 'MaskGIT':  # 🔍
        mask_ratio = schedule_maskgit_ratio(length, method=kwargs.get("mask_method", "cosine"))
        mask = [i for i in range(length) if random.random() < mask_ratio]
    elif mode == 'GPT':
        mask = [i for i in range(0, length)]
    else:
        raise ValueError("Mode should be either 'scaffolding' or 'inpainting'")

    unmask = list(set(torch.arange(length).tolist()) - set(mask))
    return sorted(unmask), sorted(mask)


def mask_data(vq_ids, mode, mask_token_id, **kwargs):
    device = vq_ids.device
    num_res = vq_ids.shape[0]

    # 🔍 for MaskGIT
    unmask_indices, mask_indices = generate_mask(num_res, mode=mode, **kwargs)

    labels = vq_ids.clone()
    vq_ids[mask_indices] = mask_token_id
    label_mask = torch.full((num_res,), fill_value=0, dtype=torch.bool, device=device)
    label_mask[mask_indices] = True
    pos_ids = torch.arange(num_res, device=device)

    return vq_ids, label_mask, pos_ids, labels


class PDBVQDataset(Dataset):
    tokenizer = PDBTokenizer()

    def __init__(
            self,
            split='train',
            max_length: int = 512,
            min_length: int = 40,
            FT=4,
            mask_method="cosine",  # 🔍
            **kwargs
    ) -> None:
        super().__init__()
        self.split = split
        self.max_length = max_length
        self.min_length = min_length
        self.MLang = MultiFoldLang()
        self.mask_method = mask_method  # 🔍

        if FT == 3:
            with open('/huyuqi/xmyu/FoldToken2/foldtoken2_data/pdb_vqids/VQDATA_vq256.jsonl', 'r', encoding='utf-8') as f:
                lines = f.readlines()

        if FT == 4:
            with open('/huyuqi/xmyu/FoldToken2/foldtoken2_data/pdb_vqids_ft4/pdb_256.jsonl', 'r', encoding='utf-8') as f:
                lines = f.readlines()

        if split == 'train':
            lines = lines[:-100]
        else:
            lines = lines[-100:]

        self.entrys = {}
        for line in lines:
            entry = json.loads(line)
            self.entrys.update(entry)
        self.names = list(self.entrys.keys())

    def __len__(self):
        return len(self.names)

    # def shuffle_chain(self, vq_ids, chain_ids):
    #     vq_ids_new = [vq_ids[0:1]]  # BOS
    #     chain_ids_new = [chain_ids[0:1]]  # BOS
    #
    #     uni_cids = torch.unique(chain_ids)
    #
    #     uni_cids = uni_cids[uni_cids < 10000]
    #     uni_cids_list = uni_cids.tolist()
    #     random.shuffle(uni_cids_list)
    #     SEP = torch.tensor([65516])
    #
    #     for i, id in enumerate(uni_cids_list):
    #         mask = chain_ids == id
    #         vq_ids_new.append(vq_ids[mask])
    #         vq_ids_new.append(SEP)
    #         chain_ids_new.append(torch.zeros_like(chain_ids[mask]) + i)
    #         chain_ids_new.append(torch.tensor([i]))
    #
    #     vq_ids_new.append(vq_ids[-1:])  # EOS
    #     chain_ids_new.append(chain_ids[-1:])  # EOS
    #     vq_ids_new = torch.cat(vq_ids_new)
    #     chain_ids_new = torch.cat(chain_ids_new)
    #     return vq_ids_new, chain_ids_new

    def __getitem__(self, index):
        try:
            entry = self.entrys[self.names[index]]

            vq_ids = torch.tensor(entry['vqid'])
            # vq_ids = self.MLang.map12to8[vq_ids]

            if entry.get('chain'):
                chain_ids = torch.tensor(entry['chain'])
            else:
                chain_ids = [1 for i in range(len(vq_ids))]

            ###### SELECT SINGLE CHAIN ######
            all_chain_lengths = torch.bincount(chain_ids)
            valid_chain_mask = (all_chain_lengths >= self.min_length) & (all_chain_lengths < self.max_length)  # 🔍 also check the max length
            chain_num = valid_chain_mask.numel()

            if valid_chain_mask.sum() == 0:
                raise RuntimeError
            else:  # there exist chains that satisfies the length limits
                valid_chain_ids = torch.arange(chain_num)[valid_chain_mask]
                cid = valid_chain_ids[random.randint(0, valid_chain_ids.numel())].item()
                chain_mask = (chain_ids == cid)

                vq_ids = vq_ids[chain_mask]
                chain_ids = chain_ids[chain_mask]
            #################################

            # Scaffold = torch.tensor([self.tokenizer.vocab['[Scaffold]']], device=device)
            # Inpaint = torch.tensor([self.tokenizer.vocab['[Inpaint]']], device=device)
            # MLM = torch.tensor([self.tokenizer.vocab['[MLM]']], device=device)
            # GPT = torch.tensor([self.tokenizer.vocab['[GPT]']], device=device)
            # ZERO = torch.tensor([0], device=device)
            # PAD = torch.tensor([self.tokenizer.vocab['[PAD]']], device=device)
            # PAD_POS = torch.tensor([1024], device=device)
            # PAD_LABEL = torch.tensor([-100], device=device)
            # Prompt = Inpaint

            vq_ids, label_mask, pos_ids, labels = mask_data(
                vq_ids,
                mode='MaskGIT',
                mask_token_id=self.tokenizer.vocab['[MLM]'],
                **{"mask_method": self.mask_method}
            )  # 🔍
            # attention mask 我改到 data interface 里面了，现在是动态 padding 到每个 batch 的最大数据长度，这样训练效率更高一些

            # length = vq_ids.shape[0]
            # vq_ids = F.pad(vq_ids, (0, self.max_length - length), value=self.tokenizer.eos_token_id)
            # label_mask = F.pad(label_mask, (0, self.max_length - length), value=False)
            # pos_ids = F.pad(pos_ids, (0, self.max_length - length), value=1024)
            # labels = F.pad(labels, (0, self.max_length - length), value=-100)

            ret = {'input_ids': vq_ids, 'label_mask': label_mask, 'pos_ids': pos_ids.long(), "labels": labels}

        except:
            ret = {'input_ids': None, 'label_mask': None, 'pos_ids': None, "labels": None}

        return ret
