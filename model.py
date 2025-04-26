import torch.nn as nn
import torch

from utils.tensor_utils import compute_angles, ortho_basis



num_emb = 6
num_dehid_pos = 4
pad_idx =5


class Block(nn.Module):
    def __init__(self, h_d, num_h, device='cuda', dropout=0.2):
        super(Block, self).__init__()
        self.device = device
        self.h_d = h_d

        self.pos_embedder = nn.Embedding(num_dehid_pos, embedding_dim=h_d)
        self.act = nn.GELU()

        self.nuk_attn = nn.MultiheadAttention(h_d, num_h, batch_first=True)
        self.nuk_attn_seq = nn.MultiheadAttention(h_d, num_h, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(h_d, h_d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_d, h_d),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(h_d)
        self.norm2 = nn.LayerNorm(h_d)
        self.norm3 = nn.LayerNorm(h_d)

        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, r, seq):
        B, L, D = seq.shape

        pos_ids = torch.arange(4, device=self.device).unsqueeze(0).repeat(B, 1)
        pos_embeds = self.pos_embedder(pos_ids)         # [B, 4, h_d]

        x = self.norm1(seq + pos_embeds)
        seq = x
        attn_out, _ = self.nuk_attn_seq(x, x, seq)
        attn_out = self.dropout_attn(attn_out)
        x = r + attn_out
        x = self.act(x)

        # --- Attention 2 (self-attn) ---
        x2 = self.norm2(x)
        attn_out, _ = self.nuk_attn(x2, x2, x2)
        attn_out = self.dropout_attn(attn_out)
        x = x + attn_out

        # --- Feedforward ---
        x3 = self.norm3(x)
        x = x + self.ffn(x3)

        return seq, x  # [B, 4, h_d]




class Transition(nn.Module):
    def __init__(self, h_d, w_d, device='cuda'):
        super(Transition, self).__init__()

        self.device = device
        self.block = nn.Sequential(nn.Linear(h_d, w_d),
                                nn.LeakyReLU(),
                                nn.Linear(w_d, w_d),
                                nn.LeakyReLU(),
                                nn.Linear(w_d, h_d)
                                )

    def forward(self, x):
        return self.block(x)





class Local_module(nn.Module):
    def __init__(self, h_d, num_nuks_head, device='cuda'):
        super(Local_module, self).__init__()

        self.device = device
        self.h_d = h_d

        self.block1 = Block(h_d, num_nuks_head, device)
        self.block2 = Block(h_d,  num_nuks_head, device)
        self.block3 = Block(h_d, num_nuks_head, device)
        self.block4 = Block(h_d, num_nuks_head, device)

        self.trans_r = Transition(h_d=h_d, w_d=2*h_d)
        self.trans_seq = Transition(h_d=h_d, w_d=2*h_d)

        self.act_fn = nn.GELU()
        self.act_tan = nn.Tanh()

        self.mean_dis = torch.tensor([1.3, 1.3, 1.3]).to(self.device)

        self.head_U = nn.Sequential(
            nn.Linear(3, 3),
            self.act_tan,
            nn.Linear(3, 3),
            self.act_tan,
            nn.Linear(3, 3),
        )

        self.act_tan = nn.Tanh()

        self.tr_layer = nn.Linear(3, 3)
        self.rad_layer = nn.Linear(3, 3)

        self.r_down = nn.Sequential(
                        nn.Linear(h_d, h_d),
                        self.act_fn,
                        nn.Linear(h_d, 32),
                        self.act_fn,
                        nn.Linear(32, 16),
                        self.act_fn,
                        nn.Linear(16, 3)
                        )

        self.r_up = nn.Sequential(
                        nn.Linear(3, 16),
                        self.act_fn,
                        nn.Linear(16, 32),
                        self.act_fn,
                        nn.Linear(32, self.h_d),
                        self.act_fn
                        )

        self.norm_r1 = nn.LayerNorm(self.h_d)
        self.norm_x1 = nn.LayerNorm(self.h_d)

        self.norm_r2 = nn.LayerNorm(self.h_d)
        self.norm_x2 = nn.LayerNorm(self.h_d)

        self.norm_r3 = nn.LayerNorm(self.h_d)
        self.norm_x3 = nn.LayerNorm(self.h_d)

        self.norm_r4 = nn.LayerNorm(self.h_d)
        self.norm_x4 = nn.LayerNorm(self.h_d)

        self.attn = nn.MultiheadAttention(embed_dim=self.h_d, num_heads=self.h_d//4, batch_first=True)
        self.norm_r = nn.LayerNorm(self.h_d)

        self.lin_r = nn.Linear(3*3, 3)
        self.lin_t = nn.Linear(3 * 3, 3)

        self.dropout_pre = nn.Dropout(p=0.2)
        self.dropout_blocks = nn.Dropout(p=0.2)
        self.dropout_attn = nn.Dropout(p=0.2)
        self.dropout_final = nn.Dropout(p=0.2)



    @staticmethod
    def inner_ort_basis(x, eps=1e-6):
        e1 = x[:, 0] / (x[:, 0].norm(dim=1, keepdim=True) + eps)
        e2 = x[:, 1] - (e1 * x[:, 1]).sum(dim=1, keepdim=True) * e1
        e2 = e2 / (e2.norm(dim=1, keepdim=True) + eps)
        e3 = x[:, 2] - (e1 * x[:, 2]).sum(dim=1, keepdim=True) * e1 - (e2 * x[:, 2]).sum(dim=1, keepdim=True) * e2
        e3 = e3 / (e3.norm(dim=1, keepdim=True) + eps)
        return torch.stack([e1, e2, e3], dim=1)  # [B, 3, 3]



    def forward(self, seq, r=None):
        batch_size = seq.size(0)

        v10 = (r[:, 1, :] - r[:, 0, :]).unsqueeze(1)
        v20 = (r[:, 2, :] - r[:, 0, :]).unsqueeze(1)
        v12 = (r[:, 1, :] - r[:, 2, :]).unsqueeze(1)
        dv = torch.cat((v10, v20, v12), dim=1)
        inv_fea = compute_angles(dv)

        vec_ortho = ortho_basis(v10.squeeze(-2), v20.squeeze(-2))
        r_ortho = self.inner_ort_basis(vec_ortho)

        inv_fea = torch.cat((inv_fea, inv_fea[:, -1, :].unsqueeze(1)), dim=1)
        inv_fea = self.r_up(inv_fea)
        inv_fea = self.dropout_pre(inv_fea)

        x, r = self.block1(inv_fea, seq)
        r_1, x_1 = self.norm_r1(r + inv_fea), self.norm_x1(x + seq)

        x, r = self.block2(r_1, x_1)
        r_2 = self.norm_r2(r_1 + r)
        x_2 = self.norm_x2(x_1 + x)

        r_2 = self.trans_r(r_2)
        x_2 = self.trans_seq(x_2)
        r_2 = self.dropout_blocks(r_2)
        x_2 = self.dropout_blocks(x_2)

        x, r = self.block3(r_2, x_2)
        x_3 = self.norm_x3(x + x_2)
        r_3 = self.norm_r3(r + r_2)

        x, r = self.block4(r_3, x_3)
        x = self.norm_x4(x + x_3)
        r = self.norm_r4(r + r_3)

        attn_out, _ = self.attn(query=x, key=x, value=r)
        attn_out = self.dropout_attn(attn_out)
        r_res = (r + attn_out)

        r = self.norm_r(r_res)
        r = self.act_fn(r)

        out = self.r_down(r)
        out = self.dropout_final(out)
        out = out[:, :-1] + out[:, -1].unsqueeze(1)

        U = self.head_U(out)

        rad_fea = self.act_fn(self.lin_r(out.reshape(batch_size, 9)).reshape(batch_size, 3))
        tr_fea = self.act_fn(self.lin_t(out.reshape(batch_size, 9)).reshape(batch_size, 3))
        rad_vec = self.rad_layer(rad_fea) + self.mean_dis
        trans_vec = self.tr_layer(tr_fea)

        I = torch.ones_like(U)
        R = r_ortho.transpose(-1, -2) @ (I + U) @ r_ortho
        rad_vec = (r_ortho.transpose(-1, -2) @ rad_vec.unsqueeze(-1)).squeeze(-1)
        trans_vec = (r_ortho.transpose(-1, -2) @ trans_vec.unsqueeze(-1)).squeeze(-1)
        rot = (R @ rad_vec.unsqueeze(-1)).squeeze(-1)

        return x, rot + trans_vec



class Global_module(nn.Module):
    def __init__(self, h_d, num_nuks_head, device='cuda'):
        super(Global_module, self).__init__()

        self.device = device

        self.h_d = h_d
        self.num_nuks_head = num_nuks_head

        self.nuk_loc_embedder = nn.Embedding(num_emb, embedding_dim=self.h_d, padding_idx=pad_idx)
        self.loc_curr = Local_module(num_nuks_head=self.num_nuks_head,  h_d=self.h_d)

        self.act_fn = nn.GELU()

        self.conv1 = nn.Conv1d(in_channels=12, out_channels=9, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=9, out_channels=6, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=6, out_channels=4, kernel_size=3)

        self.back_to_seq = nn.Linear(self.h_d - 2*3, self.h_d)


    def forward(self, seqs, r):
        seq_prev, seq_curr, seq_next = seqs[:, 0, :], seqs[:, 1, :], seqs[:, 2, :]

        batch_size = seq_prev.shape[0]

        seqs = seqs.reshape(batch_size, 12)
        seq = self.nuk_loc_embedder(seqs.long())

        seq_curr = self.act_fn(self.conv1(seq))
        seq_curr = self.act_fn(self.conv2(seq_curr))
        seq_curr = self.act_fn(self.conv3(seq_curr))

        seq_curr = self.back_to_seq(seq_curr)

        seq_curr, r_curr = self.loc_curr(seq_curr, r=r)

        r_final = r_curr

        return r_final
