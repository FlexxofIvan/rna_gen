import pandas as pd
import torch
import matplotlib.pyplot as plt
from model import Global_module
from autoreg_model import Autoreg_module
from utils.data_utils import seq_converter
from utils.eterna_utils import seq_to_bpp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r_init = torch.tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                            [ 5.4882e+00,  1.5850e+00, -1.6459e-09],
                            [ 1.0820e+01,  5.6656e-08,  1.4443e-08]]).to(device)

data_dir = 'data/test_sequences.csv'
data = pd.read_csv(data_dir)

model = Autoreg_module(gen=Global_module).to(device)
model.load_state_dict(torch.load(f'checkpoints/autoreg_epoch.pt'))

full_seq = list(data['sequence'])

def predict_r(text_seq):

    numer_seq = torch.tensor(seq_converter(text_seq), dtype=torch.float32)
    padd = torch.tensor([5, 5, 5])
    full_seq = torch.cat((padd, numer_seq, padd)).to(device)

    seq_unf = full_seq.unfold(dimension=0, size=4, step=1)
    window = [seq_unf[num] for num in range(len(seq_unf))]
    seqs = torch.stack([ torch.stack([window[num-3], window[num], window[num+3]]) for num in range(3, len(window)-3)]).to(device)
    bp = seq_to_bpp(full_seq[3:-3], numeric_repr=True).to(device)

    _, r = model(full_seq, seqs, r_init, bp)
    r = r.cpu().detach()

    return r

x = 'GGUGGCAGAGAAAGGCGAAAGCCUUGUGAGGCCAUC'
print(len(x))
r= predict_r(x)
print(r)

df = pd.DataFrame(r.numpy(), columns=["x", "y", "z"])
df.to_csv("coords.csv", index=False)

x1, y1, z1 = r[:, 0].numpy() / 100, r[:, 1].numpy() / 100, r[:, 2].numpy() / 100

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x1, y1, z1, marker='o', color='b', label="cord")

ax.set_title("3D Comparison")
ax.legend()

plt.tight_layout()
plt.show()
