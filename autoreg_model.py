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
    def __init__(self, gen, h_d, p= None, device='cuda'):
        super(Autoreg_module, self).__init__()

        self.device = device
        self.gen = gen(**loc_args)
        #self.gen.load_state_dict(torch.load(loc_weights_path))
        #self.gen.eval()

        self.p = p

        self.h_d = h_d

        self.act_fn = nn.GELU()

        self.N = 4
        self.prj_att_ln = nn.Linear(self.h_d, self.h_d)


        self.cord_prj_layer = nn.Sequential(
                                nn.Linear(self.h_d, 32),
                                self.act_fn,

                                nn.Linear(32, 16),
                                nn.LayerNorm(16),
                                self.act_fn,

                                nn.Linear(16, 8),
                                self.act_fn,

                                nn.Linear(8, 3),
                                )


        self.att_emb = nn.MultiheadAttention(self.h_d, 32, batch_first=True)
        self.norm1 = nn.LayerNorm(self.h_d)
        self.norm2 = nn.LayerNorm(self.h_d)


        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            self.act_fn,

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            self.act_fn,

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            self.act_fn,

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            self.act_fn,

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )


        self.conv_block2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            self.act_fn,

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            self.act_fn,

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            self.act_fn,

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            self.act_fn,

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        self.nuk_embedder = nn.Embedding(num_emb, embedding_dim=self.h_d, padding_idx=pad_idx)

        self.Q = nn.Sequential(
            nn.Linear(3, 8),
            self.act_fn,

            nn.Linear(8, 32),
            nn.LayerNorm(32),
            self.act_fn,

            nn.Linear(32, 64),
        )

        self.K = nn.Sequential(
            nn.Linear(3, 8),
            self.act_fn,

            nn.Linear(8, 32),
            nn.LayerNorm(32),
            self.act_fn,

            nn.Linear(32, 64),
        )

        self.V = nn.Sequential(
            nn.Linear(3, 8),
            self.act_fn,

            nn.Linear(8, 16),
            nn.LayerNorm(16),
            self.act_fn,

            nn.Linear(16, 8),
            self.act_fn,

            nn.Linear(8, 3),
        )

        self.r_norm = nn.LayerNorm(3)
        self.u_norm = nn.LayerNorm(3)

        self.denoise_layer = nn.Sequential(
            nn.Linear(3, 8),
            self.act_fn,

            nn.Linear(8, 16),
            self.act_fn,

            nn.Linear(16, 8),
            self.act_fn,

            nn.Linear(8, 3),
        )

        self.head_U = nn.Sequential(
            nn.Linear(3, 8),
            nn.LayerNorm(8),
            self.act_fn,

            nn.Linear(8, 16),
            self.act_fn,

            nn.Linear(16, 32),
            nn.LayerNorm(32),
            self.act_fn,

            nn.Linear(32, 16),
            self.act_fn,

            nn.Linear(16, 9),
        )


        self.tr_layer = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            self.act_fn,
            nn.Linear(32, 16),
            self.act_fn,
            nn.Linear(16, 3),
        )

        self.rad_layer = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            self.act_fn,
            nn.Linear(32, 16),
            self.act_fn,
            nn.Linear(16, 3),
        )

        self.init_prj = nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            self.act_fn,
            nn.Linear(16, 3),
        )

        self.dropout_attn = nn.Dropout(p=0.1)
        self.dropout_post = nn.Dropout(p=0.1)
        #self.dropout_final = nn.Dropout(p=0.1)


        self.attn = nn.MultiheadAttention(self.h_d, self.h_d//4, batch_first=True)
        self.norm = nn.LayerNorm(self.h_d)

        self.ffn = nn.Sequential(
            nn.Linear(self.h_d, self.h_d* 2),
            nn.ReLU(),
            nn.Linear(self.h_d * 2, self.h_d),
        )

        self.norm2 = nn.LayerNorm(self.h_d)

        self.ffn_r = nn.Sequential(
            nn.Linear(3, 8),
            self.act_fn,
            nn.Linear(8, 16),
            self.act_fn,
            nn.LayerNorm(16),
            nn.Linear(16, 8),
            self.act_fn,
            nn.Linear(8, 3),
        )

        self.ffn_pred = nn.Sequential(
            nn.Linear(3, 8),
            self.act_fn,
            nn.Linear(8, 16),
            self.act_fn,
            nn.LayerNorm(16),
            nn.Linear(16, 8),
            self.act_fn,
            nn.Linear(8, 3),
        )

        self.attn_pred = nn.MultiheadAttention(3, 3, batch_first=True)
        self.norm_pred = nn.LayerNorm(3)
        self.norm2_pred = nn.LayerNorm(3)

        self.init_tune = nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            self.act_fn,
            nn.Linear(16, 3),
        )


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

    @staticmethod
    def d_vecs(x):
        x = x[:, 1:] - x[:, :-1]
        return x


    @staticmethod
    def in_bas(r):

        v10 = r[:, 1, :] - r[:, 0, :]
        v20 = r[:, 2, :] - r[:, 0, :]

        vec_ortho = ortho_basis(v10, v20)
        ortho_mat = inner_ort_basis(vec_ortho)
        return  ortho_mat



    def forward(self, full_seq, seqs, init_deltas, bps, sch_samp=False, targets=None):

        emb =  self.nuk_embedder(full_seq.long())  + self.gen.nuk_loc_embedder(full_seq.long())

       # emb = self.apply_rope(emb).to(self.device)

        b_s, seq_len, _ = emb.shape
        bps = self.conv_block(bps.float().unsqueeze(1)).squeeze(1)


        #emb_mat = torch.einsum('ijk, ikl -> ijl', emb, torch.permute(emb, (0, 2, 1)))
        bps = bps #+ emb_mat
        bps = bps.to(torch.float32) #+ emb_mat

        attn_output, attn_logits = self.att_emb(emb, emb, emb, need_weights=True)
        attn_logits = attn_logits + bps
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_output = torch.matmul(attn_weights, emb)

        emb = self.norm1(attn_output)
        emb = self.dropout_attn(emb)

        residual = emb
        output = self.prj_att_ln(attn_output)
        output = output + residual
        output = self.act_fn(output)

        attn_out, _ = self.attn(output, output, output)
        x = self.norm(output + attn_out)
        ffn_out = self.ffn(x)
        output = self.norm2(x + ffn_out)

        output = self.act_fn(output)
        output = self.dropout_post(output)  # Dropout перед предсказанием координат


        init_cord = init_deltas + self.init_prj(init_deltas)

        dr = self.cord_prj_layer(output)

        U_pred = self.head_U(dr).reshape(b_s, seq_len, 3, 3)[:, :-3]
        rad_pred = self.rad_layer(dr)[:, :-1]
        tr_pred = self.tr_layer(dr)[:, :-1]


        r_curr = init_cord.to(self.device)


        cord = torch.empty((b_s, 0, 3)).to(self.device)
        loc = torch.empty((b_s, 0, 3)).to(self.device)


        init_sh = self.d_vecs(init_cord)
        zeros = torch.zeros((b_s, 1, 3)).to(self.device)
        loc = torch.cat([loc, zeros, init_sh], dim=1)
        cord = torch.cat([cord, init_cord], dim=1)


        trans_mat = torch.empty((b_s, 0, 3, 3)).to(self.device)


        I = (torch.eye(3).repeat(b_s, 1, 1)).to(self.device)
        I = I.unsqueeze(1)
        I = I.repeat(1, 3, 1, 1)
        trans_mat = torch.cat([trans_mat, I], dim=1)



        for num in range(max_len-3):

            if sch_samp:
                r_curr_tr = targets[:, num:num + 3]
                if torch.rand(1).item() <= self.p :
                    r_curr = r_curr_tr


            v10 = r_curr[:, 1, :] - r_curr[:, 0, :]
            v20 = r_curr[:, 2, :] - r_curr[:, 0, :]

            vec_ortho = ortho_basis(v10, v20)
            ortho_mat = inner_ort_basis(vec_ortho)
            trans_mat = torch.cat([trans_mat, ortho_mat.unsqueeze(1)], dim=1)

            r_pred, local_vec = self.gen(seqs[:, num], r=r_curr)

            I = torch.eye(3).repeat(b_s, 1, 1).to(self.device)

            A = U_pred[:, num]

            r_pred = torch.einsum('lkj, lj -> lk', I+A, local_vec + rad_pred[:, num]) + tr_pred[:, num]
            r_pred = r_pred  + self.ffn_pred(r_pred)

            loc = torch.cat([loc, r_pred.unsqueeze(1)], dim=1)
            r_pred = torch.einsum('lkj, lj -> lk', ortho_mat.transpose(-1, -2), r_pred)

            #r_pred = torch.einsum('lkj, lj -> lk', ortho_mat.transpose(-1,-2), r_pred)
            r_pred = r_curr[:, -1] + r_pred
            cord = torch.cat([cord, r_pred.unsqueeze(1)], dim=1)

            r_curr = torch.cat([r_curr[:, 1:], r_pred.unsqueeze(1)], dim=1)


        bps =  self.conv_block2(bps.unsqueeze(1)).squeeze(1)
        loc = self.ffn_r(loc)

        #attn_out, attn_weights = self.attn_pred(loc, loc, loc)  # (B, T, 3)

        attn_output, attn_weights = self.attn_pred(loc, loc, loc, need_weights=True)
        attn_weights = bps+attn_weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = loc + attn_output


        loc = self.norm_pred(loc + attn_out)  # Residual + norm

        # Feed-forward
        ffn_out = self.ffn_pred(loc)  # (B, T, 3)
        loc = loc + self.norm2_pred(ffn_out)


        dcord = torch.einsum('ijkl, ijk -> ijl', trans_mat, loc)
        cord = cord #+ dcord

        return cord




