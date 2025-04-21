import torch
from torch.nn import L1Loss

from model import Global_module
from autoreg_model import Autoreg_module, N
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt

data_dir = 'data/data_filt_autoreg.pt'

data = torch.load(data_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modela = Autoreg_module(gen=Global_module).to(device)

optimizer = optim.Adam(
    modela.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

def vis_two(r1, r2, loss=None, step=None):
    x1, y1, z1 = r1[:, 0].numpy()/100, r1[:, 1].numpy()/100, r1[:, 2].numpy()/100
    x2, y2, z2 = r2[:, 0].numpy()/100, r2[:, 1].numpy()/100, r2[:, 2].numpy()/100

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x1, y1, z1, marker='o', color='b', label="cord")
    ax.plot(x2, y2, z2, marker='x', color='r', label="r_tar")

    # Добавим loss и step
    title = "3D Comparison"
    if loss is not None or step is not None:
        title += " |"
        if step is not None:
            title += f" Step: {step}"
        if loss is not None:
            title += f" | Loss: {loss:.4f}"

    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.show()



### добавим полную последовательность в фичи
full_data = []
for num in range(len(data)):
    seqs, r_fea, bp, r_tar = data[num]

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


    full_data.append((full_seq, seqs, r_fea, bp, r_tar))


modela.load_state_dict(torch.load(f'checkpoints/autoreg_epoch.pt'))
import random
criterion = nn.SmoothL1Loss(beta=5) #50
batch_size = 32
#random.shuffle(full_data)

indices_to_drop = {198, 362}
filtered_data = [item for i, item in enumerate(full_data) if i not in indices_to_drop]

for epoch in range(10000):
    total_loss = 0.0
    count = 0
    batch_loss = 0.0
    batch_count = 0

    optimizer.zero_grad()
    random.shuffle(full_data)
    for i, (full_seqs, seqs, r_fea, bp, r_tar) in enumerate(full_data):
        if full_seqs.shape[0] == 0 or seqs.shape[0] in [1, 2]:
            continue

        seqs = seqs.to(device)
        r_fea = r_fea.to(device)
        r_tar = r_tar.to(device)
        bp = bp.to(device)

        noise = 0.1*torch.randn_like(r_fea)
        diff_pred, r_pred = modela(full_seqs, seqs, r_fea+noise, bp)

        r_tar = r_tar
        diff = r_tar.unsqueeze(1) - r_tar.unsqueeze(0)


        loss = (criterion(diff, diff_pred) + criterion(r_tar, r_pred)) # возможно тут надо заменить срез
        #if loss>5:
         #   continue
        batch_loss += loss
        batch_count += 1

        if batch_count == batch_size:
            avg_batch_loss = batch_loss / batch_count
            print(avg_batch_loss)
            #if avg_batch_loss>4.5:
              #  continue

            #vis_two(r_pred.detach().cpu(), r_tar.cpu(), loss = avg_batch_loss, step =i)

            avg_batch_loss.backward()
            optimizer.step()
            #if epoch==0:
               # for name, param in modela.named_parameters():
                #    if param.grad is not None:
                 #       print(f"{name}: grad shape {param.grad.shape}, grad mean {param.grad.mean().item():.4f}")
                  #  else:
                   #     print(f"{name}: grad is None")

            optimizer.step()
            optimizer.zero_grad()

            total_loss += avg_batch_loss.item()
            count += 1

            batch_loss = 0.0
            batch_count = 0


    # если остался "хвост" в батче
    if batch_count > 0:
        avg_batch_loss = batch_loss / batch_count
        avg_batch_loss.backward()
        optimizer.step()
        total_loss += avg_batch_loss.item()
        count += 1

    avg_loss = total_loss / count if count > 0 else 0
    print(f"[Epoch {epoch + 1}] Average Loss: {avg_loss:.6f}")
    torch.save(modela.state_dict(), f'checkpoints/autoreg_epoch.pt')