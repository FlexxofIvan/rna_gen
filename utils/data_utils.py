import pandas as pd
import torch
import numpy as np
from constants import nuks_val, rev_nuks_val


def group_by_rna(path: str):
    """"
    get df with rnas, grouping it
    """
    df = pd.read_csv(path)
    df = df.dropna()
    df['gr_id'] = (df['resid']==1).cumsum()
    groups = [x for _,x in list(df.groupby("gr_id"))]
    return groups


def seq_converter(seq, reverse=False):
    """
    convert chain to num or num to chain
    """
    if reverse:
        return list(map(lambda x: rev_nuks_val[int(x)], seq))
    return list(map(lambda x: nuks_val[x], seq))


pad_symb = 'p'
pad_num = nuks_val[pad_symb]

def select_seq_and_cord(groups, seq_num: False, padding:False):
    """
    groups: df of rnas
    select seq and coords from dataframe
    """
    data = []
    if padding:
        padding_tensor = torch.zeros(4, 3)
        #pad_symb = rev_nuks_val[pad_sym]

    for group in groups:
        r = torch.tensor(np.array(group.loc[:, ['x_1', 'y_1', 'z_1']]), dtype=torch.float32)
        if padding:
            r = torch.cat((padding_tensor, r, padding_tensor), dim=0)

        seq = list(group['resname'])
        if padding:
            seq = (4*pad_symb + ''.join(seq) + 4*pad_symb)
            seq = [x for x in seq]

        if seq_num:
            seq = torch.tensor(seq_converter(seq), dtype=torch.float32)
        data.append((seq, r))

    return data



def unfold_seq_r(datas, padding: False):
    """
    make windows with 4 nuks for coordinates and seqs
    """
    seqs, rs = datas
    if padding:
        r_unfs = rs.unfold(dimension=0, size=4, step=1).transpose(-2, -1)[1:-1]
        seq_unfs = seqs.unfold(dimension=0, size=4, step=1)[1:-1]
    else:
        r_unfs = rs.unfold(dimension=0, size=4, step=1).transpose(-2, -1)
        seq_unfs = seqs.unfold(dimension=0, size=4, step=1)

    return seq_unfs, r_unfs





