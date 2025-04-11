from torch.utils.data import Dataset


class nuks_seq_dataset(Dataset):
    def __init__(self, dataset):
        self.X = [(seq, first_r) for ((seq, first_r), _) in dataset]
        self.y = [lst_r for ((_, _), lst_r) in dataset]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]