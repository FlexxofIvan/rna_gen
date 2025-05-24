import torch

from model import Global_module
from autoreg_model import Autoreg_module, N

import torch.optim as optim
import torch.nn as nn

from utils.tensor_utils import loc_basis
import matplotlib.pyplot as plt
import random
from data_prep.batched_dataset import device, train_loader, test_loader
import torch.nn.utils as nn_utils


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



model = Autoreg_module(gen=Global_module, h_d=64, p=1.0).to(device)

#trainable_params = filter(lambda p: p.requires_grad, modela.parameters())

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


model.load_state_dict(torch.load(f'checkpoints/autoreg_epoch.pt'))
criterion = nn.L1Loss()


def d_vecs(x):
    x = x[:,1:] - x[:,:-1]
    return x


#M = 20


def train(data_train, data_val, model, loss_fn, epoch_num):

    for epoch in range(epoch_num):
        total_loss = 0
        optimizer.zero_grad()
        model.train()
        for i, batch in enumerate(data_train):

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
            r_fea = r_fea #+noise
            #sch_samp=True, targets=r_tar
            r_pred = model(full_seqs, seqs, r_fea, bp, sch_samp=True, targets=r_tar)
            #r_tar = r_tar[:, :50]
            #r_tar = r_tar
            batch_size = r_pred.shape[0]

            #r_pred = r_pred[:, :49]

            mask = (torch.norm(r_tar, dim=-1) != 0).float()

            mask[:, 0] = 1.0

            r_tar = r_tar
            r_pred = r_pred

            diff_pred = r_pred.unsqueeze(2) - r_pred.unsqueeze(1)
            diff_tar = r_tar.unsqueeze(2) - r_tar.unsqueeze(1)

            #r_tar = r_tar - r_tar[:, 0]
            delta_tar = d_vecs(r_tar) #[:, :M]

            delta_pred = d_vecs(r_pred) #[:, :M] #d_vecs(r_pred)

            delta_mask = mask[:, 1:] * mask[:, :-1]  # [B, L-1]
            delta_mask = delta_mask.unsqueeze(-1) #[:, :M]

            delta_pred = delta_pred * delta_mask
            delta_tar = delta_tar * delta_mask
            delta_diff = delta_pred - delta_tar

            delta_diff_masked = delta_diff
            delta_loss = (torch.norm(delta_diff_masked, dim=-1).sum()) / (delta_mask).sum()

            pair_mask = (mask.unsqueeze(1) * mask.unsqueeze(2))
            pair_loss = torch.norm(diff_pred - diff_tar, p=1, dim=-1)

            pair_loss = pair_loss
            pair_mask = pair_mask

            pair_loss = (pair_loss * pair_mask).sum() / (pair_mask.sum() + 1e-6)

            loss = 0.1*pair_loss #pair_loss #delta_loss#pair_loss #delta_loss  +pair_loss


            if torch.isnan(loss):
                print("⚠️ Loss is NaN! Пропускаем итерацию или завершаем обучение.")
                optimizer.zero_grad()
                continue

            print(loss)


            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            """
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
                else:
                    print(f"{name}: grad is None")
            """

            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(data_train)
        print(f"[Epoch {epoch + 1}] Train loss: {avg_train_loss:.4f}")

        torch.save(model.state_dict(), f'checkpoints/autoreg_epoch.pt')

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

                r_pred = model(full_seqs, seqs, r_fea, bp)

                r_pred = r_pred
                r_tar = r_tar
                batch_size = r_pred.shape[0]

                mask = (torch.norm(r_tar, dim=-1) != 0).float()
                mask[:, 0] = 1.0


                delta_pred = d_vecs(r_pred)  # d_vecs(r_pred)
                delta_tar = d_vecs(r_tar)

                delta_mask = mask[:, 1:] * mask[:, :-1]  # [B, L-1]
                delta_mask = delta_mask.unsqueeze(-1)

                delta_pred = delta_pred * delta_mask
                delta_tar = delta_tar * delta_mask
                delta_diff = delta_pred - delta_tar

                delta_diff_masked = delta_diff

                delta_loss = (torch.norm(delta_diff_masked, dim=-1).sum()) / delta_mask.sum()

                loss = delta_loss
                #pair_loss = ((torch.abs(diff_pred - diff_tar) * pair_mask).sum()) / pair_mask.sum()

                #loss = pair_loss
                val_loss += loss.item()

        val_loss /= len(data_val)
        print(f"[Epoch {epoch + 1}] Val loss: {val_loss:.4f}")

        model.train()


train(train_loader, test_loader, model, criterion, 1000)