import torch
from torch.nn import L1Loss

from model import Global_module
from autoreg_model import Autoreg_module
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

data_dir = 'data/data_filt_autoreg.pt'

data = torch.load(data_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoreg_module(gen=Global_module).to(device)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-5
)

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


import random
criterion = nn.MSELoss()
batch_size = 32
random.shuffle(full_data)

for epoch in range(100):
    total_loss = 0.0
    count = 0
    batch_loss = 0.0
    batch_count = 0

    optimizer.zero_grad()

    for i, (full_seqs, seqs, r_fea, bp, r_tar) in enumerate(full_data):
        if full_seqs.shape[0] == 0 or seqs.shape[0] in [1, 2]:
            continue

        seqs = seqs.to(device)
        r_fea = r_fea.to(device)
        r_tar = r_tar.to(device)
        bp = bp.to(device)

        r_pred = model(full_seqs, seqs, r_fea, bp)

        loss = criterion(r_pred, r_tar[:5])  # возможно тут надо заменить срез
        batch_loss += loss
        batch_count += 1

        if batch_count == batch_size:
            avg_batch_loss = batch_loss / batch_count
            print(avg_batch_loss)
            avg_batch_loss.backward()
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
        optimizer.zero_grad()

        total_loss += avg_batch_loss.item()
        count += 1

    avg_loss = total_loss / count if count > 0 else 0
    print(f"[Epoch {epoch + 1}] Average Loss: {avg_loss:.6f}")