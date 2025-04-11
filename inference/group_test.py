import torch
import matplotlib.pyplot as plt
from model import Global_module
from autoreg_model import Autoreg_module

data_dir = '../data/data_filt_autoreg.pt'


data = torch.load(data_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoreg_module(gen=Global_module).to(device)


seqs, r_fea, bp, r_tar = data[19]

seqs = seqs.to(device)
r_init = r_fea.to(device)
r_tar = r_tar.to(device)

r = model.autoreg_gen(seqs, r_fea).cpu()



def vis_two(r1, r2):
    x1, y1, z1 = r1[:, 0].numpy(), r1[:, 1].numpy(), r1[:, 2].numpy()
    x2, y2, z2 = r2[:, 0].numpy(), r2[:, 1].numpy(), r2[:, 2].numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x1, y1, z1, marker='o', color='b', label="cord")
    ax.plot(x2, y2, z2, marker='x', color='r', label="r_tar")

    ax.set_title("3D Comparison")
    ax.legend()

    plt.tight_layout()
    plt.show()


vis_two(r, r_tar.cpu())



