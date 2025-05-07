import torch
from utils.tensor_utils import loc_basis
from torch.nn.utils.rnn import pad_sequence
from constants import nuks_val, max_len
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split





data = torch.load("data/data_filt_autoreg.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

means_init = torch.tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                            [ 5.4882e+00,  1.5850e+00, -1.6459e-09],
                            [ 1.0820e+01,  5.6656e-08,  1.4443e-08]])

padd_symb = nuks_val['p']


def custom_collate(batch):
    full_seqs_list, seqs_list, means_list, bp_list, r_tar_list = zip(*batch)

    def tens_pad(fea, padd_val):
        padd_size = max_len - fea.shape[0]
        tens_size = fea.shape[1:]
        padd = torch.full((padd_size,)+tens_size, padd_val).to(device)
        fea = torch.cat([fea, padd], dim=0)
        return fea

    def bpp_pad(fea):
        padd_size = max_len - fea.shape[0]

        padd_vert = torch.zeros((padd_size, fea.shape[1])).to(device)
        fea = torch.cat((fea, padd_vert), dim=0)

        padd_hor = torch.zeros((fea.shape[0], padd_size)).to(device)
        fea = torch.cat((fea, padd_hor), dim=1)

        return fea


    full_seqs = torch.stack([tens_pad(x,  padd_symb) for x in full_seqs_list])
    seqs = torch.stack([tens_pad(x,  padd_symb) for x in seqs_list])
    r_tar = torch.stack([tens_pad(x,  0) for x in r_tar_list])
    bpps = torch.stack([bpp_pad(x) for x in bp_list])
    means = torch.stack(means_list)

    return {
        'full_seq': full_seqs,
        'seqs': seqs,
        'means_init': means,
        'bp': bpps,
        'r_tar': r_tar
    }


full_data = []
for num in range(len(data)):
    seqs, _, bp, r_tar = data[num]
    full_seq = torch.empty(0).to(device)
    if seqs.shape[0] == 1:
        full_seq = seqs[0]
    for part_num, seq in enumerate(seqs.to(device)):
        if part_num == 0:
            full_seq = seq[0]
        elif part_num == len(seqs)-1:
            prev = seq[0][-1].unsqueeze(0)
            curr = seq[1][1:-1]
            next = seq[-1]
            last_seq = torch.cat([prev, curr, next])
            full_seq = torch.cat([full_seq, last_seq])
        else:
            full_seq = torch.cat([full_seq, seq[0][-1].unsqueeze(0)]).to(device)

    r_tar = r_tar - r_tar[0]
    dv = r_tar[:3]
    R1 = loc_basis(dv)
    r_tar = torch.einsum('ij, lj -> li', R1.transpose(-2, -1), r_tar)
    diff = r_tar[1:] - r_tar[:-1]
    norm = torch.norm(diff, dim=-1)
    if (norm > 10).any():
        pass
    else:
        ###
        #padding do 200
        ###
        full_data.append((full_seq, seqs.to(device), means_init.to(device), bp.to(device), r_tar.to(device)))


class MyRNADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset = MyRNADataset(full_data)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=custom_collate
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=custom_collate)