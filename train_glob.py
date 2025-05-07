import torch

from model import Global_module
from autoreg_model import Autoreg_module, N

import torch.optim as optim
import torch.nn as nn

from utils.tensor_utils import loc_basis
import matplotlib.pyplot as plt
import random
from data_prep.batched_dataset import device, train_loader, test_loader


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



modela = Autoreg_module(gen=Global_module, h_d=64).to(device)

#trainable_params = filter(lambda p: p.requires_grad, modela.parameters())

optimizer = optim.Adam(
    modela.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


#modela.load_state_dict(torch.load(f'checkpoints/autoreg_epoch.pt'))
criterion = nn.L1Loss()


def d_vecs(x):
    x = x[:,1:] - x[:,:-1]
    x = torch.cat((x[:,0].unsqueeze(1), x), dim=1)
    return x


def train(data_train, data_val, model, loss_fn, epoch_num):

    for epoch in range(epoch_num):

        total_loss = 0

        optimizer.zero_grad()
        for batch in data_train:
            full_seqs = batch['full_seq']
            seqs = batch['seqs']
            r_fea = batch['means_init']
            bp = batch['bp']
            r_tar = batch['r_tar']


            seqs = seqs.to(device)
            r_fea = r_fea.to(device)
            r_tar = r_tar.to(device)
            bp = bp.to(device)

            noise = 0.2*torch.randn_like(r_fea)
            r_fea = (r_fea+noise)
            r_pred = model(full_seqs, seqs, r_fea, bp)

            #r_tar = r_tar
            mask = (((torch.norm(r_tar, dim=-1) != 0).float()).reshape(8, 128, 1)).repeat(1, 1, 3)
            diff_pred = r_pred.unsqueeze(2) - r_pred.unsqueeze(1)

            r_pred = mask*r_pred

            r_tar = r_tar - r_tar[0].unsqueeze(0)
            diff = r_tar.unsqueeze(2) - r_tar.unsqueeze(1)

            diff_r = d_vecs(r_tar)
            diff_model = d_vecs(r_pred)

            loss = (loss_fn(diff, diff_pred) + loss_fn(r_tar, r_pred) + loss_fn(diff_r, diff_model))/3 # возможно тут надо заменить срез

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(data_train)
        print(f"[Epoch {epoch + 1}] Train loss: {avg_train_loss:.4f}")

        #torch.save(modela.state_dict(), f'checkpoints/autoreg_epoch.pt')

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in data_val:
                full_seqs = batch['full_seq']
                seqs = batch['seqs']
                r_fea = batch['means_init']
                bp = batch['bp']
                r_tar = batch['r_tar']

                seqs = seqs.to(device)
                r_fea = r_fea.to(device)
                r_tar = r_tar.to(device)
                bp = bp.to(device)

                diff_pred, r_pred = model(full_seqs, seqs, r_fea, bp)

                r_tar = r_tar - r_tar[0].unsqueeze(0)
                diff = r_tar.unsqueeze(2) - r_tar.unsqueeze(1)

                diff_r = d_vecs(r_tar)
                diff_model = d_vecs(r_pred)

                loss = (loss_fn(diff, diff_pred) + loss_fn(r_tar, r_pred) + loss_fn(diff_r, diff_model)) / 3
                val_loss += loss.item()

        val_loss /= len(data_val)
        print(f"[Epoch {epoch + 1}] Val loss: {val_loss:.4f}")

        model.train()


train(train_loader, test_loader, modela, criterion, 1000)