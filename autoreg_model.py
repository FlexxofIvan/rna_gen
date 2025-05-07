import torch.nn as nn
import torch
import os

from utils.tensor_utils import ortho_basis, inner_ort_basis, mat_mul_vec
import torch.nn.functional as F
import math
from constants import max_len

loc_weights_path ='checkpoints/nuk_4nn.pt'


loc_args = {'h_d': 64,
            'num_nuks_head': 32,
            'device': 'cuda'
            }


N = 50

num_emb = 6
pad_idx =5


class Autoreg_module(nn.Module):
    def __init__(self, gen, h_d, device='cuda'):
        super(Autoreg_module, self).__init__()

        self.device = device
        self.gen = gen(**loc_args)
        #self.gen.load_state_dict(torch.load(loc_weights_path))
        #self.gen.eval()
        #for param in self.gen.parameters():
         #   param.requires_grad = False


        self.h_d = h_d

        self.act_fn = nn.GELU()

        self.N = 4
        self.K_lin = nn.Linear(self.h_d, self.N*self.h_d)
        self.Q_lin = nn.Linear(self.h_d, self.N*self.h_d)
        self.V_lin = nn.Linear(self.h_d, self.N * self.h_d)

        self.prj_att_ln = nn.Linear(self.h_d, self.h_d)

        self.cord_prj_layer = nn.Sequential(
                                nn.Linear(self.h_d, 32),
                                nn.LayerNorm(32),
                                self.act_fn,
                                nn.Dropout(0.1),
                                nn.Linear(32, 16),
                                nn.LayerNorm(16),
                                self.act_fn,
                                nn.Linear(16, 8),
                                nn.LayerNorm(8),
                                self.act_fn,
                                nn.Linear(8, 3),
                                nn.LayerNorm(3),
                                )

        self.att_emb = nn.MultiheadAttention(self.h_d, 32, batch_first=True)
        self.norm1 = nn.LayerNorm(self.h_d)
        self.norm2 = nn.LayerNorm(self.h_d)

        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(4, affine=True),
                                        self.act_fn,
                                        nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(16, affine=True),
                                        self.act_fn,
                                        nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(4, affine=True),
                                        self.act_fn,
                                        nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1),
                                        )


        self.nuk_embedder = nn.Embedding(num_emb, embedding_dim=self.h_d, padding_idx=pad_idx)

        self.Q = nn.Sequential(nn.Linear(3, 8),
                                self.act_fn,
                                nn.Linear(8, 32),
                                self.act_fn,
                                nn.Linear(32, 64)
                                )

        self.K = nn.Sequential(nn.Linear(3, 8),
                               self.act_fn,
                               nn.Linear(8, 32),
                               self.act_fn,
                               nn.Linear(32, 64)
                               )
        self.V = nn.Linear(3, 3)

        self.r_norm = nn.LayerNorm(3)
        self.u_norm = nn.LayerNorm(3)

        self.denoise_layer = nn.Sequential(nn.Linear(3, 8),
                                           self.act_fn,
                                           nn.Linear(8, 3)
                                           )

        self.act_tan = nn.Tanh()

        self.head_U = nn.Sequential(
            nn.Linear(3, 5),
            self.act_tan,
            nn.Linear(5, 7),
            self.act_tan,
            nn.Linear(7, 9),
        )


        self.U_init = nn.Sequential(
            nn.Linear(3, 5),
            self.act_tan,
            nn.Linear(5, 7),
            self.act_tan,
            nn.Linear(7, 9),
        )

        self.rad_init = nn.Linear(3, 3)

        self.trans_init = nn.Linear(3, 3)

        self.tr_layer = nn.Linear(3, 3)
        self.rad_layer = nn.Linear(3, 3)

        self.init_prj = nn.Linear(3, 3)

        self.dropout_attn = nn.Dropout(p=0.3)
        self.dropout_post = nn.Dropout(p=0.3)
        self.dropout_init = nn.Dropout(p=0.3)
        self.dropout_final = nn.Dropout(p=0.3)


    @staticmethod
    def apply_rope(x):

        B, L, D = x.shape
        assert D % 2 == 0, "Embedding dim must be even"

        x = x.view(B, L, D // 2, 2)  # → (B, L, D/2, 2)

        pos = torch.arange(L, device=x.device, dtype=torch.float32)  # (L,)
        dim = torch.arange(D // 2, device=x.device, dtype=torch.float32)  # (D/2,)
        freqs = 1.0 / (10000 ** (dim / (D // 2)))  # (D/2,)
        angles = torch.einsum("l,d->ld", pos, freqs)  # (L, D/2)

        cos = torch.cos(angles)[None, :, :, None]  # (1, L, D/2, 1)
        sin = torch.sin(angles)[None, :, :, None]  # (1, L, D/2, 1)

        x_rotated = torch.cat([
            x[..., 0:1] * cos - x[..., 1:2] * sin,
            x[..., 0:1] * sin + x[..., 1:2] * cos
        ], dim=-1)

        return x_rotated.view(B, L, D)



    def forward(self, full_seq, seqs, init_deltas, bps):

        emb = self.gen.nuk_loc_embedder(full_seq.long()) + self.nuk_embedder(full_seq.long())
        emb = self.apply_rope(emb).to(self.device)

        b_s, seq_len, _ = emb.shape

        emb_mat = torch.einsum('ijk, ikl -> ijl', emb, torch.permute(emb, (0, 2, 1)))
        bps = emb_mat * bps
        bps = bps.to(torch.float32)
        bps = self.conv_block(bps.unsqueeze(1)).squeeze(1)

        residual = emb
        attn_output, attn_weights = self.att_emb(emb, emb, emb, need_weights=True)
        attn_weights = attn_weights * bps
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        attn_output = torch.matmul(attn_weights, emb)

        emb = self.norm1(residual + attn_output)
        emb = self.dropout_attn(emb)

        residual = emb
        output = self.prj_att_ln(attn_output)
        output = self.norm2(output + residual)
        output = self.act_fn(output)

        output = self.act_fn(output)
        output = self.dropout_post(output)  # Dropout перед предсказанием координат

        dr = self.cord_prj_layer(output)

        dr_init = self.act_fn(self.init_prj(dr[:,:3]))
        U_init = self.act_fn(self.U_init(dr_init).reshape(-1, 3, 3, 3))
        rad_init = self.act_fn(self.rad_init(dr_init).reshape(-1, 3, 3))
        trans_init = self.act_fn(self.trans_init(dr_init).reshape(-1, 3, 3))

        I_init = torch.ones_like(U_init)
        init_cord = torch.einsum('sljk,slk-> slj', (I_init + U_init), rad_init) + trans_init
        init_cord = init_deltas + self.denoise_layer(init_cord)

        dr_pred = dr[:, 3:]
        U_pred = self.head_U(dr_pred).reshape(b_s, seq_len-3, 3, 3)
        I = torch.ones_like(U_pred)

        rad_pred = self.rad_layer(dr_pred)
        tr_pred = self.tr_layer(dr_pred)
        mean = torch.tensor([[1, 1.5, 1.3]]).to(self.device).repeat(1, seq_len-3, 1)
        dr_pred = rad_pred + mean
        dr_pred = self.u_norm(dr_pred)

        r_pred = torch.einsum('sijk,sik->sij', (I + U_pred), dr_pred)
        dr_pred = r_pred + tr_pred


        r_curr = init_cord.to(self.device)
        cord = torch.empty((b_s, 0, 3)).to(self.device)
        shifts_pred = torch.empty((b_s, 0, 3)).to(self.device)
        full_cords_pred = torch.empty((b_s, 0, 3)).to(self.device)

        cord = torch.cat([cord, init_cord], dim=1)
        for num in range(max_len-3):
            v10 = r_curr[:, 1, :] - r_curr[:, 0, :]
            v20 = r_curr[:, 2, :] - r_curr[:, 0, :]

            r_pred = self.gen(seqs[:, num], r=r_curr)

            delta = dr_pred[:, num]
            vec_ortho = ortho_basis(v10, v20)
            ortho_mat = inner_ort_basis(vec_ortho)

            shifts_pred = torch.cat([shifts_pred, r_pred.unsqueeze(1)], dim=1)

            r_pred = torch.einsum('lkj, lj -> lk', ortho_mat.transpose(-1, -2) ,delta) + r_curr[:, -1] + r_pred
            full_cords_pred = torch.cat([full_cords_pred, r_pred.unsqueeze(1)], dim=1)

            r_curr = torch.cat([r_curr[:, 1:], r_pred.unsqueeze(1)], dim=1)
            cord = torch.cat([cord, r_pred.unsqueeze(1)], dim=1)


        diff = cord.unsqueeze(2) - cord.unsqueeze(1)

        diff_q = self.Q(diff)
        diff_k = self.K(diff).permute(0, 3, 2, 1)
        dr_v = self.V(dr)
        mat = torch.einsum('sijk, skjl->sil', diff_q, diff_k)
        mat = self.dropout_final(mat)
        d_k = diff_q.shape[-1]
        mat = mat / (d_k ** 0.5)

        attn_weights = torch.softmax(mat, dim=-1)
        df = torch.einsum('sij,sjl->sil', attn_weights, dr_v)

        df = self.act_fn(df)
        df = self.r_norm(df)

        cord = df + cord

        return cord




