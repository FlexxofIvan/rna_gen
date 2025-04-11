from rdkit import Chem
import numpy as np
import os
import torch
from dataset import nuks_seq_dataset
from constants import nuks_val


dir_path = 'dihedrals'

def get_coords_dehid(dir_name, nuk_name):
    path = f'{dir_name}/{nuk_name}.sdf'
    supplier = Chem.SDMolSupplier(path)
    dehid_arr = []
    for mol in supplier:
        if mol is not None:
            conformer = mol.GetConformer()
            nuks_coord = np.empty((0, 3))
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                pos = conformer.GetAtomPosition(idx)
                nuks_coord = np.concatenate([ nuks_coord, torch.tensor([[pos.x, pos.y, pos.z]], dtype=torch.float32) ], axis=0)
            dehid_arr.append(nuks_coord)
    return dehid_arr



dataset = []
for name in os.listdir(dir_path):
    coords = get_coords_dehid(dir_path, name[:-4])
    code = torch.tensor([nuks_val[let] for let in name[:-4]], dtype=torch.float32)
    X = [((code, r[:-1, :]), r[-1, :]) for r in coords]
    dataset.extend(X)


DATASET = nuks_seq_dataset(dataset)


"""
train_size = int(0.8 * len(DATASET))  # 80% - train, 20% - test
test_size = len(DATASET) - train_size
train_dataset, test_dataset = random_split(DATASET, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

"""