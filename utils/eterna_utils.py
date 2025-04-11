import os
os.environ["ARNIEFILE"] = f"/home/ivan/anaconda3/envs/mols/lib/python3.10/site-packages/arnie/arnie.txt"

import torch
from utils.data_utils import seq_converter
from arnie.bpps import bpps
import matplotlib.pyplot as plt
import seaborn as sns



def seq_to_bpp(seq, numeric_repr=False):

    if not numeric_repr:
        return torch.tensor(bpps(seq, package="eternafold"))

    if numeric_repr:
        seq_let = seq_converter(seq, reverse=True)
        seq_str = ''.join(seq_let)
        return torch.tensor(bpps(seq_str, package="eternafold"))




def plot_bpp_heatmap(bpp_matrix, sequence=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(bpp_matrix, cmap="viridis", square=True, cbar=True, linewidths=0.1, linecolor='gray')

    if sequence:
        labels = list(sequence)
        plt.xticks(ticks=range(len(sequence)), labels=labels)
        plt.yticks(ticks=range(len(sequence)), labels=labels)

    plt.title("Base Pairing Probability Matrix (BPP)")
    plt.xlabel("Nucleotide Position")
    plt.ylabel("Nucleotide Position")
    plt.tight_layout()
    plt.show()


