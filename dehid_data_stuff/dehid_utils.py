from constants import (rev_nuks_val, nuks_val, means_norm_val,
                       means_ang_val, means_ortho_val)
from data_utils import data
import torch


def build_lookup_from_dict(d: dict):
    index_keys = []
    index_vals = []

    for k, v in d.items():
        key_tensor = torch.tensor(k, dtype=torch.long)  # без nuks_val
        index_keys.append(key_tensor)
        index_vals.append(v.unsqueeze(0))  # (1, 3, 3)

    lookup_keys = torch.stack(index_keys)         # (N, 4)
    lookup_vals = torch.cat(index_vals, dim=0)    # (N, 3, 3)

    return lookup_keys, lookup_vals



def dict_search(batch_x: torch.Tensor, lookup_keys: torch.Tensor, lookup_vals: torch.Tensor, device='cuda', ortho=False):
    """
    batch_x: (batch_size, 3, 4) — нуклеотидные последовательности
    lookup_keys: (N, 4)
    lookup_vals: (N, 3)

    returns: (batch_size, 3, 3) — нормы и углы
    """
    bsz, num_seq, seq_len = batch_x.shape
    flat_x = batch_x.view(-1, seq_len).to(device)  # (B*3, 4)
    lookup_keys = lookup_keys.to(device)
    lookup_vals = lookup_vals.to(device)

    matches = (flat_x.unsqueeze(1) == lookup_keys.unsqueeze(0))  # (B*3, N, 4)
    matched = matches.all(dim=-1)  # (B*3, N)

    has_match = matched.any(dim=1)
    matched_idxs = matched.float().argmax(dim=1)

    rand_idxs = torch.randint(0, lookup_vals.size(0), (flat_x.size(0),), device=device)

    final_idxs = torch.where(has_match, matched_idxs, rand_idxs)

    out = lookup_vals[final_idxs]  # (B*3, 3)

    if not ortho:
        return out.view(bsz, num_seq, 3)
    else:
        return out.view(bsz, num_seq, 3, 3)



def inv_fea_enc(x, lk_keys_norm, lk_keys_ang, lk_vals_ang, lk_vals_norm):

    out_ang = dict_search(x, lk_keys_ang, lk_vals_ang, ortho=False).unsqueeze(2)
    out_norm = dict_search(x, lk_keys_norm, lk_vals_norm, ortho=False).unsqueeze(2)

    b_size,_,_,_ = out_norm.shape

    I = torch.ones_like(out_ang)
    sin = (I - out_ang**2)**0.5

    cos_prj = out_norm*out_ang
    sin_prj = sin*out_norm

    return torch.cat([out_norm, cos_prj, sin_prj], dim=2)


