from utils.data_utils import group_by_rna, select_seq_and_cord, unfold_seq_r
from data_prep.dataset import nuks_seq_dataset
import torch


###padding sucks

train_groups = group_by_rna('data/train_labels.csv')
data = select_seq_and_cord(train_groups, True, padding=False) ### take seqs and coordinates

window_lst = []
for num in range(1, len(data)-1):
    seq, _ = data[num]

    seq_len = seq.shape[0]  ###нужно чтоб билось на окна
    if seq_len < 4:
        continue

    seq_unf, r_unf = unfold_seq_r(data[num], padding=False)

    new_data = [( seq_unf[num], r_unf[num])
            for num in range(len(seq_unf))]
    window_lst.extend(new_data)



data_glob = [(torch.stack([window_lst[num-3][0], window_lst[num][0], window_lst[num+3][0]]), window_lst[num][1])
     for num in range(3, len(window_lst)-3)]

DATASET = [((seq, x[:-1]), x[-1]) for seq,x in data_glob if torch.norm(x[:-1][-1]-x[-1]) < 10.0]


dataset_loc = nuks_seq_dataset(DATASET)