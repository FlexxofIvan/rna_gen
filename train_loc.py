import torch
import matplotlib.pyplot as plt
from model import Global_module
from data_prep.make_dataset_loc import dataset_loc

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import torch.nn.utils as nn_utils

from utils.tensor_utils import compute_angles, ortho_basis
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


### блять отфильтрованы только тары, иниты нет, ограничение на норму не снимать
data_dir = 'data/data_filt_autoreg.pt'


data = torch.load(data_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loc_args = {'h_d': 32,
            'num_nuks_head': 32,
            'device': 'cuda'
            }
model = Global_module(**loc_args).to(device)


BATCH_SIZE = 128
EPOCHS = 10000
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Деление на train/test
full_dataset = dataset_loc
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

#model.load_state_dict(torch.load(f'checkpoints/nuk_4nn.pt'))

train_losses = []
val_losses = []


def inner_ort_basis(x, eps=1e-6):
    e1 = x[:, 0] / (x[:, 0].norm(dim=1, keepdim=True) + eps)
    e2 = x[:, 1] - (e1 * x[:, 1]).sum(dim=1, keepdim=True) * e1
    e2 = e2 / (e2.norm(dim=1, keepdim=True) + eps)
    e3 = x[:, 2] - (e1 * x[:, 2]).sum(dim=1, keepdim=True) * e1 - (e2 * x[:, 2]).sum(dim=1, keepdim=True) * e2
    e3 = e3 / (e3.norm(dim=1, keepdim=True) + eps)
    return torch.stack([e1, e2, e3], dim=1)  # [B, 3, 3]



for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):

        (seqs, init_cord), targets = batch
        seqs, init_cord, targets = seqs.to(DEVICE), init_cord.to(DEVICE), targets.to(DEVICE)

        noise = torch.randn_like(init_cord)

        r = init_cord + noise

        v10 = (r[:, 1, :] - r[:, 0, :]).unsqueeze(1)
        v20 = (r[:, 2, :] - r[:, 0, :]).unsqueeze(1)
        v12 = (r[:, 1, :] - r[:, 0, :]).unsqueeze(1)

        vec_ortho = ortho_basis(v10.squeeze(-2), v20.squeeze(-2))
        r_ortho = inner_ort_basis(vec_ortho)

        rev_mat = r_ortho

        taros = targets - init_cord[:, -1]

        dv = torch.einsum('ijk, ik -> ij', r_ortho, taros)


        L1 = torch.norm(init_cord[:, 1] - init_cord[:, 0], dim=-1)
        L2 = torch.norm(init_cord[:, 2] - init_cord[:, 1], dim=-1)

        
        if max(L1)>10 or max(L2)>10 :
            continue


        optimizer.zero_grad()
        init_cord = init_cord
        outputs, rad_vec = model(seqs, init_cord)

        loss = criterion(rad_vec, dv)  + criterion(outputs, taros)

        if torch.isnan(loss):
            print(seqs, init_cord, targets)
        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=1)

        """
        if epoch == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
                else:
                    print(f"{name}: grad is None")
        """

        optimizer.step()
        loss_ap = criterion(outputs, targets-init_cord[:, -1])
        train_loss += loss_ap.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Валидация
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
            (seqs, init_cord), targets = batch
            seqs, init_cord, targets = seqs.to(DEVICE), init_cord.to(DEVICE), targets.to(DEVICE)

            outputs, _ = model(seqs, init_cord)

            taros = targets - init_cord[:, -1]


            L1 = torch.norm(init_cord[:, 1] - init_cord[:, 0], dim=-1)
            L2 = torch.norm(init_cord[:, 2] - init_cord[:, 1], dim=-1)

            if max(L1) > 10 or max(L2) > 10:
                continue


            loss = criterion(outputs, taros)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(test_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    torch.save(model.state_dict(), f'checkpoints/nuk_4nn.pt')

# Визуализация
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid()
plt.show()
