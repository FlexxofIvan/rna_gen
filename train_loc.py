import torch.optim as optim
import torch
from model import Global_module
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from data_prep.data_utils import DATASET



# === Разделение данных на train/test ===
train_size = int(0.8 * len(DATASET))  # 80% - train, 20% - test
test_size = len(DATASET) - train_size
train_dataset, test_dataset = random_split(DATASET, [train_size, test_size])

train_loader = DataLoader(DATASET, batch_size=32, shuffle=True)
test_loader = DataLoader(DATASET, batch_size=32, shuffle=True)

# === Инициализация модели и оптимизатора ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Global_module(h_d=64, num_nuks_head=32).to(device)

model.load_state_dict(torch.load("nuk_4_nn.pth"))

mean_delta = torch.tensor(1.25)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)


num_epochs = 500


def random_transform(r):
    batch_size, seq_len, dim = r.shape
    Q, _ = torch.linalg.qr(torch.randn(batch_size, dim, dim, device=r.device))
    t = torch.randn(batch_size, 1, dim, device=r.device) * 5
    return Q, t


for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    means = []

    for batch_idx, (inputs, target_r) in enumerate(train_loader):
        (seqs, r_init) = inputs
        seqs = seqs.to(device).float()
        r_init = r_init.to(device).float()
        target_r = target_r.to(device).float()

        optimizer.zero_grad()
        b_size, _, _ = r_init.shape
        Q, t = random_transform(r_init)

        r = torch.cat((r_init, target_r.unsqueeze(1)), dim=1)
        r_init = r[:, :3, :]
        target_r = r[:, -1, :]

        #pred_r = model(seqs,  r=r_init)
        y = target_r - r_init[:, -1, :]

        noise_scale = 0.5  #
        r_init_noisy = r_init

        pred_r = model(seqs, r=r_init_noisy)


        means.append(torch.norm(y, dim=-1).mean())
        loss = criterion(pred_r, y)
        loss.backward()
        optimizer.step()

        total_train_loss += criterion(pred_r, y).item()

    print(f"Epoch {epoch} finished. Avg Loss: {total_train_loss / len(train_loader):.6f}")




    # === Тестирование ===
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for (seqs, r_init), target_r in test_loader:
            seqs, r_init, target_r = seqs.to(device).float(), r_init.to(device).float(), target_r.to(device).float()

            b_size, _, _ = r_init.shape
            Q, t = random_transform(r_init)

            r = torch.cat((r_init, target_r.unsqueeze(1)), dim=1)
            r  = torch.einsum('bij,bnj->bni', Q, r) + t

            r_init = r[:, :3, :]
            target_r = r[:, -1, :]

            pred_r = model(seqs, r=r_init)

            y = target_r - r_init[:, -1, :]
            test_loss = criterion(pred_r, y)
            total_test_loss += test_loss.item()



    # === Логирование ===
    avg_train_loss = total_train_loss / len(train_loader)
    avg_test_loss = total_test_loss / len(test_loader)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

    # Сохранение модели
    torch.save(model.state_dict(), "nuk_4_nn.pth")