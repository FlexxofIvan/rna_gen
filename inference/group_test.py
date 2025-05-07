import torch
import matplotlib.pyplot as plt
from model import Global_module
from autoreg_model import Autoreg_module
from utils.tensor_utils import loc_basis



 ### подгружаем данные
data_dir = '../data/data_filt_autoreg.pt'
data = torch.load(data_dir)

### средние начальные координаты
means_init = torch.tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                            [ 5.4882e+00,  1.5850e+00, -1.6459e-09],
                            [ 1.0820e+01,  5.6656e-08,  1.4443e-08]])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoreg_module(gen=Global_module).to(device)


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
    ###по сути выравнил начальные 3 нуклеотида поворотом
    r_tar = r_tar - r_tar[0]
    dv = r_tar
    R1 = loc_basis(dv)
    r_tar = torch.einsum('ij, lj -> li', R1.transpose(-2, -1), r_tar)
    full_data.append((full_seq, seqs, means_init, bp, r_tar))

full_seq, seqs, r_fea, bp, r_tar = full_data[47]

full_seq = full_seq.to(device)
seqs = seqs.to(device)
r_init = r_fea.to(device)
r_tar = r_tar.to(device)
bp= bp.to(device)

r_tar = r_tar - r_tar[0]

model.load_state_dict(torch.load(f'../checkpoints/autoreg_epoch.pt'))
model.eval()
_, r = model(full_seq, seqs, r_init, bp)
r = r.detach().cpu()


def vis_two(r1, r2):
    x1, y1, z1 = r1[:, 0].numpy()/100, r1[:, 1].numpy()/100, r1[:, 2].numpy()/100
    x2, y2, z2 = r2[:, 0].numpy()/100, r2[:, 1].numpy()/100, r2[:, 2].numpy()/100

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x1, y1, z1, marker='o', color='b', label="cord")
    ax.plot(x2, y2, z2, marker='x', color='r', label="r_tar")

    ax.set_title("3D Comparison")
    ax.legend()

    plt.tight_layout()
    plt.show()


vis_two(r, r_tar.cpu())


