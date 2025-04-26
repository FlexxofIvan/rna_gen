import torch.nn as nn
import torch
import os

from utils.tensor_utils import ortho_basis, inner_ort_basis, mat_mul_vec
import torch.nn.functional as F
import math

#root_dir = os.path.dirname(os.path.abspath(__file__))
loc_weights_path ='../checkpoints/nuk_4nn.pt'


loc_args = {'h_d': 64,
            'num_nuks_head': 32,
            'device': 'cuda'
            }


N = 20

num_emb = 6
pad_idx =5


class Autoreg_module(nn.Module):
    def __init__(self, gen, device='cuda'):
        super(Autoreg_module, self).__init__()

        self.device = device
        self.gen = gen(**loc_args)
        self.gen.load_state_dict(torch.load(loc_weights_path))
        self.gen.eval()


        self.act_fn = nn.GELU()

        self.N = 4
        self.K_lin = nn.Linear(self.gen.h_d, self.N*self.gen.h_d)
        self.Q_lin = nn.Linear(self.gen.h_d, self.N*self.gen.h_d)
        self.V_lin = nn.Linear(self.gen.h_d, self.N * self.gen.h_d)

        self.prj_att_ln = nn.Linear(self.N*self.gen.h_d, self.gen.h_d)

        self.cord_prj_layer = nn.Sequential(
                                nn.Linear(self.gen.h_d, 32),
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

        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                                        nn.InstanceNorm2d(4, affine=True),
                                        self.act_fn,
                                        nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
                                        nn.InstanceNorm2d(16, affine=True),
                                        self.act_fn,
                                        nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1),
                                        nn.InstanceNorm2d(4, affine=True),
                                        self.act_fn,
                                        nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1),
                                        )


        self.nuk_embedder = nn.Embedding(num_emb, embedding_dim=self.gen.h_d, padding_idx=pad_idx)

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

        self.conv = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=0)

        self.init_prj = nn.Linear(3, 3)

        self.dropout_emb = nn.Dropout(p=0.2)
        self.dropout_attn = nn.Dropout(p=0.2)
        self.dropout_post = nn.Dropout(p=0.2)
        self.dropout_init = nn.Dropout(p=0.2)
        self.dropout_final = nn.Dropout(p=0.2)



    def apply_rope(self, x, theta):

        x1, x2 = x[..., ::2], x[..., 1::2]
        sin_theta, cos_theta = theta[..., ::2].to(self.device), theta[..., 1::2].to(self.device)

        x_rotated = torch.cat([
            x1 * cos_theta - x2 * sin_theta,  # Новая x-компонента
            x1 * sin_theta + x2 * cos_theta  # Новая y-компонента
        ], dim=-1)

        return x_rotated


    def forward(self, full_seq, seqs, init_deltas, bps):
        emb = self.gen.nuk_loc_embedder(full_seq.long()) + self.nuk_embedder(full_seq.long())
        seq_len, _ = emb.shape

        padd_vert = torch.zeros((6, bps.shape[1])).to(self.device)
        bps = torch.cat((bps, padd_vert), dim=0)
        padd_hor = torch.zeros((bps.shape[0], 6)).to(self.device)
        bps = torch.cat((bps, padd_hor), dim=1)

        emb_mat = emb @ emb.transpose(-2, 1)

        pos = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb.size(1), 2) * -(math.log(10000.0) / emb.size(1)))
        pe = torch.zeros(seq_len, emb.size(1))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pos_emb = self.apply_rope(emb, pe).to(self.device)

        emb = pos_emb + emb
        emb = self.dropout_emb(emb)  # Dropout после объединения positional + token embeddings

        bps = emb_mat * bps
        q = self.Q_lin(emb).reshape(seq_len, self.N, self.gen.h_d).permute(1, 0, 2)
        k = self.K_lin(emb).reshape(seq_len, self.N, self.gen.h_d).permute(1, 0, 2)
        v = self.V_lin(emb).view(seq_len, self.N, self.gen.h_d).permute(1, 0, 2)

        bps = bps.to(torch.float32)
        bps = self.conv_block(bps.unsqueeze(0)).mean(0)

        score = torch.matmul(q, k.transpose(-2, -1)) / (float(self.gen.h_d) ** 0.5)
        bps = bps.unsqueeze(0).repeat(self.N, 1, 1)
        score = F.softmax(score, dim=-1).to(torch.float32)
        attn_weights = bps + score
        attn_output = torch.matmul(attn_weights, v).permute(1, 0, 2).reshape(seq_len, self.N * self.gen.h_d)

        attn_output = self.dropout_attn(attn_output)  # Dropout после внимания
        output = emb + self.prj_att_ln(self.act_fn(attn_output))
        output = self.act_fn(output)
        output = self.dropout_post(output)  # Dropout перед предсказанием координат

        dr = self.cord_prj_layer(output)
        dr = self.conv(torch.permute(dr, (1, 0)))
        dr = torch.permute(dr, (1, 0))

        mean = torch.tensor([[0.4, 0.5, 0.2]]).to(self.device).repeat(seq_len, 1)[6:-3]

        dr_init = self.act_fn(self.init_prj(dr[:3]))
        dr_init = self.dropout_init(dr_init)  # Dropout перед геометрией инициализации

        U_init = self.act_fn(self.U_init(dr_init).reshape(3, 3, 3))
        rad_init = self.act_fn(self.rad_init(dr_init).reshape(3, 3))
        trans_init = self.act_fn(self.trans_init(dr_init).reshape(3, 3))

        I_init = torch.ones_like(U_init)
        init_cord = torch.einsum('ljk,lk->lj', (I_init + U_init), rad_init) + trans_init
        init_cord = init_deltas + self.denoise_layer(init_cord)

        dr_pred = dr[3:]
        U_pred = self.head_U(dr_pred).reshape(-1, 3, 3)
        I = torch.ones_like(U_pred)

        rad_pred = self.rad_layer(dr_pred)
        tr_pred = self.tr_layer(dr_pred)
        dr_pred = rad_pred + mean
        dr_pred = self.u_norm(dr_pred)

        r_pred = torch.einsum('ijk,ik->ij', (I + U_pred), dr_pred)
        dr_pred = r_pred + tr_pred

        r_curr = init_cord.to(self.device)
        cord = torch.empty((0, 3)).to(self.device)
        shifts_pred = torch.empty((0, 3)).to(self.device)
        full_cords_pred = torch.empty((0, 3)).to(self.device)

        cord = torch.cat([cord, init_cord])
        for num, seq in enumerate(seqs):
            v10 = r_curr[1, :] - r_curr[0, :]
            v20 = r_curr[2, :] - r_curr[0, :]
            r_pred = self.gen(seq.unsqueeze(0), r=r_curr.unsqueeze(0))

            delta = dr_pred[num]
            vec_ortho = ortho_basis(v10.unsqueeze(0), v20.unsqueeze(0))
            ortho_mat = inner_ort_basis(vec_ortho)

            shifts_pred = torch.cat([shifts_pred, r_pred], dim=0)
            r_pred = ortho_mat.transpose(-1, -2) @ delta + r_curr[-1] + r_pred
            full_cords_pred = torch.cat([full_cords_pred, r_pred], dim=0)

            r_curr = torch.cat([r_curr[1:], r_pred], dim=0)
            cord = torch.cat([cord, r_pred])

        diff = cord.unsqueeze(1) - cord.unsqueeze(0)
        diff_q = self.Q(diff)
        diff_k = self.K(diff).permute(2, 1, 0)
        dr_v = self.V(dr)
        mat = torch.einsum('ijk,kjl->il', diff_q, diff_k)
        mat = self.dropout_final(mat)
        d_k = diff_q.shape[-1]
        mat = mat / (d_k ** 0.5)

        attn_weights = torch.softmax(mat, dim=-1)
        df = torch.einsum('ij,jl->il', attn_weights, dr_v)

        df = self.act_fn(df)
        df = self.r_norm(df)

        cord = df + cord
        return diff, cord




