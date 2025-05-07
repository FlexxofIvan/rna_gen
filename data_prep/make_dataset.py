from utils.data_utils import group_by_rna, select_seq_and_cord, unfold_seq_r
from utils.eterna_utils import seq_to_bpp
import torch


###padding sucks

data_dir = '../data/train_labels.csv'
train_groups = group_by_rna(data_dir)
data = select_seq_and_cord(train_groups, True, padding=True) ### take seqs and coordinates


window_lst = []
for num in range(0, len(data)):
    seq, _ = data[num]
    seq_len = seq.shape[0]  ###нужно чтоб билось на окна
    if seq_len < 4:
        continue
    seq_unf, r_unf = unfold_seq_r(data[num], padding=True)
    new_data = [( seq_unf[num], r_unf[num])
            for num in range(len(seq_unf))]
    window_lst.append(new_data)


data_test = []
for num, window in enumerate(window_lst):
    seqs = torch.empty(0, 3, 4)
    tar_r = torch.empty(0, 3)
    init_r = torch.empty(0, 3, 3)

    full_seq = torch.empty(0)
    for part_num, (seq, _) in enumerate(window):
        if part_num == 0:
            full_seq = seq
        else:
            full_seq = torch.cat([full_seq, seq[-1].unsqueeze(0)])


    bpp_seq = seq_to_bpp(full_seq[3:-3], numeric_repr=True)
    data_glob = [(torch.stack([window[num-3][0], window[num][0], window[num+3][0]]), window[num][1])
         for num in range(3, len(window)-3)]


    for number, (seq, r) in enumerate(data_glob):

        if number == 0:
            tar_r = torch.cat((tar_r, r), dim=0)
            init_r = r[:-1]
        else:
            tar_r = torch.cat((tar_r, r[-1].unsqueeze(0)))

        seqs= torch.cat((seqs, seq.unsqueeze(0)), dim=0)

    data_test.append((seqs, init_r, bpp_seq, tar_r))


data_filt = [data for data in data_test if 0<data[0].shape[0]<150]
#data_filt = [data for data in data_test if 70<data[0].shape[0]]
#print(len(data_filt))
data_filt_autoreg = [(seq, r_init[0], bpp, r_tar) for (seq, r_init, bpp, r_tar) in data_filt]

#torch.save(data_filt_autoreg, "data_filt_autoreg.pt")