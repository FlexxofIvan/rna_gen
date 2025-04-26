import torch
import matplotlib.pyplot as plt
from model import Global_module
from data_prep.make_dataset_loc import dataset_loc

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


data_dir = 'data/data_filt_autoreg.pt'


data = torch.load(data_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loc_args = {'h_d': 64,
            'num_nuks_head': 32,
            'device': 'cuda'
            }
model = Global_module(**loc_args).to(device)


BATCH_SIZE = 128
EPOCHS = 1000
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Деление на train/test
full_dataset = dataset_loc
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

model.load_state_dict(torch.load(f'checkpoints/nuk_4nn.pt'))

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):

        (seqs, init_cord), targets = batch
        seqs, init_cord, targets = seqs.to(DEVICE), init_cord.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        noise = 0.2 * torch.randn_like(init_cord)
        init_cord = init_cord + noise
        outputs = model(seqs, init_cord)
        loss = criterion(outputs, targets-init_cord[:, -1])
        if torch.isnan(loss):
            print(seqs, init_cord, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Валидация
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
            (seqs, init_cord), targets = batch
            seqs, init_cord, targets = seqs.to(DEVICE), init_cord.to(DEVICE), targets.to(DEVICE)

            outputs = model(seqs, init_cord)
            loss = criterion(outputs, targets - init_cord[:, -1])
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
