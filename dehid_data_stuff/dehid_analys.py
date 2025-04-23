import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from rdkit import Chem
import pandas as pd
import seaborn as sns
import torch
import numpy as np
from constants import nuks_val
import os


dir_path = '../dihedrals'

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


def compute_angles_norms(x: torch.tensor, epsilon=1e-6):
    norms = torch.norm(x, dim=-1, keepdim=True)
    unit_vectors = x / (norms + epsilon)

    cos_angles = torch.einsum("bij,bij->bi", unit_vectors[:, [0, 1, 0]], unit_vectors[:, [1, 2, 2]]).unsqueeze(
        dim=-1)

    return cos_angles


def d3_visual(r: torch.tensor) -> None:
    x, y, z = r[:, 0], r[:, 1], r[:, 2]

    x = x.numpy()
    y = y.numpy()
    z = z.numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, marker='o', color='b', label="Seq Coordinates")
    plt.show()


dataset = []
means_ang = []
stds_ang = []
Y_means = []
Y_stds = []
names = []

def compute_vector_metrics(x: torch.tensor, epsilon=1e-6):
    norms = torch.norm(x, dim=-1, keepdim=True)  # (б_с, 3, 1)

    unit_vectors = x / (norms + epsilon)

    cos10 = torch.einsum('bi,bi->b', unit_vectors[:, 1], unit_vectors[:, 0]).unsqueeze(1)
    cos20 = torch.einsum('bi,bi->b', unit_vectors[:, 0], unit_vectors[:, 2]).unsqueeze(1)
    cos12 = torch.einsum('bi,bi->b', unit_vectors[:, 2], unit_vectors[:, 1]).unsqueeze(1)
    cos_angl = torch.cat((cos10, cos20, cos12), dim=1).unsqueeze(1)

    I = torch.ones_like(cos_angl)

    sin_angl = (I - cos_angl**2)**0.5

    norms = norms.reshape(-1, 1, 3)

    cos_norm = norms*cos_angl
    sin_norm = norms*sin_angl

    res = torch.cat((norms, cos_norm, sin_norm), dim=1)
    return res


def ortho_basis(x, y):
    r = torch.stack([x, y, torch.cross(x, y)], dim=1)
    return r


for name in os.listdir(dir_path):
    coords = get_coords_dehid(dir_path, name[:-4])

    X = [torch.tensor(r[:-1, :]) for r in coords]

    d_tens = torch.empty((0,3,3))
    for x in coords:
        r = torch.tensor(x[:-1, :])
        dv = r - r[0].unsqueeze(0)
        d_tens = torch.cat((d_tens, dv.unsqueeze(0)), dim=0)
    mean_init_cord = d_tens.mean(0)
    #x = compute_vector_metrics(d_tens)
    #X = torch.stack(X)

    #Y = [ortho_basis(torch.tensor(x[1]-x[0]),  torch.tensor(x[2]-x[0]))
      #   for x in [r[:-1, :] for r in coords]]

   # Y = torch.stack(Y)

    #Y_mean = torch.mean(Y, dim=0)
    #Y_std = torch.std(Y, dim=0)

    #mean_ang_val = torch.mean(X, dim=0)
    #std_ang_val = torch.std(X, dim=0)

    #means_ang.append(mean_ang_val)
    #stds_ang.append(std_ang_val)

    #Y_means.append(Y_mean)
    #Y_stds.append(Y_std)

    names.append(name[:-4])
    dataset.append(mean_init_cord)



means_dict = {name: y for name,y in zip(names, dataset)}


#means_ang_val = {name: y for name,y in zip(names, means_ang)}

def save_dict_to_py(dict_name, data_dict, encoder_dict, filename="../constants.py"):
    with open(filename, "a") as f:
        f.write(f"{dict_name} = {{\n")
        for key_str, tensor in data_dict.items():
            # Преобразуем строковый ключ ('ACGU') в кортеж индексов (0, 2, 3, 1)
            key_idx = tuple(encoder_dict[char] for char in key_str)
            f.write(f"    {key_idx}: torch.tensor({tensor.tolist()}),\n")
        f.write("}\n")

save_dict_to_py("means_dict", means_dict, nuks_val)
#save_dict_to_py("means_ang_val", means_ang_val, nuks_val)

### means stds visual
"""
for num in range(len(dataset)):
    points = dataset[num]

    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(points)

    x = points_2d[:, 0]
    y = points_2d[:, 1]

    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=x, y=y, fill=True, cmap="viridis")
    plt.title('KDE Plot of Dihedral Angles (PCA-Reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
"""