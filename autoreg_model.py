import torch.nn as nn
import torch
import os

from utils.tensor_utils import ortho_basis, inner_ort_basis, mat_mul_vec
import torch.nn.functional as F
import math

root_dir = os.path.dirname(os.path.abspath(__file__))
loc_weights_path = os.path.join(root_dir, 'nuk_4_nn.pth')


loc_args = {'h_d': 64,
            'num_nuks_head': 32,
            'device': 'cuda'
            }


N = 10


class Autoreg_module(nn.Module):
    def __init__(self, gen, device='cuda'):
        super(Autoreg_module, self).__init__()

        self.device = device
        self.gen = gen(**loc_args)
        self.gen.load_state_dict(torch.load(loc_weights_path))
        #self.gen.eval()
        #for param in self.gen.parameters():
        #    param.requires_grad = False

        self.act_fn = nn.GELU()

        self.N = 4
        self.K_lin = nn.Linear(self.gen.h_d, self.N*self.gen.h_d)
        self.Q_lin = nn.Linear(self.gen.h_d, self.N*self.gen.h_d)
        self.V_lin = nn.Linear(self.gen.h_d, self.N * self.gen.h_d)

        self.prj_att_ln = nn.Linear(self.N*self.gen.h_d, self.gen.h_d)
        self.norm = nn.LayerNorm(self.gen.h_d)

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
                                       # nn.InstanceNorm2d(1, affine=True),
                                        #nn.Sigmoid(),
                                        )

        self.proj_prev = nn.Sequential(nn.Linear(3, 8),
                                       self.act_fn,
                                       nn.Linear(8, 16),
                                       self.act_fn,
                                       nn.Linear(16, 16),
                                       self.act_fn,
                                       nn.Linear(16, 8),
                                       self.act_fn,
                                       nn.Linear(8, 3)
                                        )

        self.nuk_embedder = nn.Embedding(5, embedding_dim=self.gen.h_d)

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

        self.tr_layer = nn.Linear(3, 3)
        self.rad_layer = nn.Linear(3, 3)

        self.noise_norm = nn.LayerNorm(3)

    def apply_rope(self, x, theta):

        x1, x2 = x[..., ::2], x[..., 1::2]
        sin_theta, cos_theta = theta[..., ::2].to(self.device), theta[..., 1::2].to(self.device)

        x_rotated = torch.cat([
            x1 * cos_theta - x2 * sin_theta,  # Новая x-компонента
            x1 * sin_theta + x2 * cos_theta  # Новая y-компонента
        ], dim=-1)

        return x_rotated


    def forward(self, full_seq, seqs, init_cord, bps):

        emb = self.gen.nuk_loc_embedder(full_seq.long()) + self.nuk_embedder(full_seq.long())
        seq_len, _ = emb.shape

        emb_mat = emb@emb.transpose(-2,1)

        pos = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb.size(1), 2) * -(math.log(10000.0) / emb.size(1)))
        pe = torch.zeros(seq_len, emb.size(1))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pos_emb = self.apply_rope(emb, pe).to(self.device)

        emb = pos_emb+emb
        bps = emb_mat * bps

        q = self.Q_lin(emb).reshape(seq_len, self.N, self.gen.h_d).permute(1, 0, 2)
        k = self.K_lin(emb).reshape(seq_len, self.N, self.gen.h_d).permute(1, 0, 2)
        v = self.V_lin(emb)
        v = v.view(seq_len, self.N, self.gen.h_d).permute(1, 0, 2)

        bps = bps.to(torch.float32)
        bps = self.conv_block(bps.unsqueeze(0)).mean(0)

        score = torch.matmul(q, k.transpose(-2, -1))  # [N, L, L]
        score = score / (float(self.gen.h_d))**0.5
        bps = bps.unsqueeze(0).repeat(self.N, 1, 1)
        score = score.to(torch.float32)
        score = F.softmax(score, dim=-1).to(torch.float32)  # [N, L, L]
        attn_weights = bps + score
        attn_output = torch.matmul(attn_weights, v).permute(1, 0, 2).reshape(seq_len, self.N*self.gen.h_d)
        output = emb + self.prj_att_ln(self.act_fn(attn_output))
        output = self.act_fn(output)

        #print(output)
        dr = self.cord_prj_layer(output)[6:-3] #на 9 больше, тк первые шесть не предсказываем и последние три тоже


        mean = torch.tensor([[0.4, 0.5, 0.2]]).to(self.device).repeat(seq_len, 1)[6:-3]

        denoised_out = self.noise_norm(self.denoise_layer(init_cord))
        init_cord = init_cord + denoised_out

        U = self.head_U(dr).reshape(-1, 3, 3)
        I = torch.ones_like(U)
        r = self.rad_layer(dr)
        tr = self.tr_layer(dr)
        dr = r + mean
        dr = self.u_norm(dr)
        r = torch.einsum('ijk,ik->ij', (I+U), dr)
        dr = r+tr
        r_curr = init_cord.to(self.device)
        cord = torch.empty((0, 3)).to(self.device)
        shifts_pred = torch.empty((0, 3)).to(self.device)
        full_cords_pred = torch.empty((0, 3)).to(self.device)


        for num, seq in enumerate(seqs):
            v10 = (r_curr[1, :] - r_curr[0, :])
            v20 = (r_curr[2, :] - r_curr[0, :])
            delta = dr[num]
            vec_ortho = ortho_basis(v10.unsqueeze(0), v20.unsqueeze(0))              # получаем матрицу перехода которую используем в изначальном генераторе
            ortho_mat = inner_ort_basis(vec_ortho) #.transpose(-1, -2)             #чтобы было согласованно с предсказанием

            r_pred = self.gen(seq.unsqueeze(0), r=r_curr.unsqueeze(0))
            shifts_pred = torch.cat([shifts_pred, r_pred], dim=0)

            #I = torch.ones_like(U)
            #R = ortho_mat.transpose(-1, -2) @ (I + U) @ ortho_mat
            #rad_vec = (ortho_mat.transpose(-1, -2) @ rad_vec.unsqueeze(-1)).squeeze(-1)
            #trans_vec = (ortho_mat.transpose(-1, -2) @ trans_vec.unsqueeze(-1)).squeeze(-1)

            r_pred = ortho_mat.transpose(-1, -2)@delta + r_curr[-1] + r_pred
            full_cords_pred = torch.cat([full_cords_pred, r_pred], dim=0)

            r_curr = torch.cat([r_curr[1:], r_pred], dim=0)
            cord = torch.cat([cord, r_pred])
        diff = cord.unsqueeze(1) - cord.unsqueeze(0)

        diff_q = self.Q(diff)
        diff_k = self.K(diff).permute(2, 1, 0)
        dr_v = self.V(dr)
        mat = torch.einsum('ijk,kjl->il', diff_q, diff_k)
        d_k = diff_q.shape[-1]
        mat = mat / (d_k ** 0.5)
        attn_weights = torch.softmax(mat, dim=-1)
        df = torch.einsum('ij,jl->il', attn_weights, dr_v)

        df = self.act_fn(df)
        df = self.r_norm(df)

        cord = df+cord

        return diff, cord


    def full_gen(self, full_seq, seqs, init_cord, bps):

        emb = self.gen.nuk_loc_embedder(full_seq.long()) + self.nuk_embedder(full_seq.long())
        seq_len, _ = emb.shape

        emb_mat = emb @ emb.transpose(-2, 1)

        pos = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb.size(1), 2) * -(math.log(10000.0) / emb.size(1)))
        pe = torch.zeros(seq_len, emb.size(1))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pos_emb = self.apply_rope(emb, pe).to(self.device)

        emb = pos_emb + emb
        bps = emb_mat * bps

        q = self.Q_lin(emb).reshape(seq_len, self.N, self.gen.h_d).permute(1, 0, 2)
        k = self.K_lin(emb).reshape(seq_len, self.N, self.gen.h_d).permute(1, 0, 2)
        v = self.V_lin(emb)
        v = v.view(seq_len, self.N, self.gen.h_d).permute(1, 0, 2)

        bps = bps.to(torch.float32)
        bps = self.conv_block(bps.unsqueeze(0)).mean(0)

        score = torch.matmul(q, k.transpose(-2, -1))  # [N, L, L]
        score = score / (float(self.gen.h_d)) ** 0.5
        bps = bps.unsqueeze(0).repeat(self.N, 1, 1)
        score = score.to(torch.float32)
        score = F.softmax(score, dim=-1).to(torch.float32)  # [N, L, L]
        attn_weights = bps + score
        attn_output = torch.matmul(attn_weights, v).permute(1, 0, 2).reshape(seq_len, self.N * self.gen.h_d)
        output = emb + self.prj_att_ln(self.act_fn(attn_output))
        output = self.act_fn(output)

        # print(output)
        dr = self.cord_prj_layer(output)[6:-3]  # на 9 больше, тк первые шесть не предсказываем и последние три тоже

        mean = torch.tensor([[0.4, 0.5, 0.2]]).to(self.device).repeat(seq_len, 1)[6:-3]

        denoised_out = self.noise_norm(self.denoise_layer(init_cord))
        init_cord = init_cord + denoised_out

        U = self.head_U(dr).reshape(-1, 3, 3)
        I = torch.ones_like(U)
        r = self.rad_layer(dr)
        tr = self.tr_layer(dr)
        dr = r + mean
        dr = self.u_norm(dr)
        r = torch.einsum('ijk,ik->ij', (I + U), dr)
        dr = r + tr
        r_curr = init_cord.to(self.device)
        cord = torch.empty((0, 3)).to(self.device)
        shifts_pred = torch.empty((0, 3)).to(self.device)
        full_cords_pred = torch.empty((0, 3)).to(self.device)

        for num, seq in enumerate(seqs):
            v10 = (r_curr[1, :] - r_curr[0, :])
            v20 = (r_curr[2, :] - r_curr[0, :])
            delta = dr[num]
            vec_ortho = ortho_basis(v10.unsqueeze(0), v20.unsqueeze(
                0))  # получаем матрицу перехода которую используем в изначальном генераторе
            ortho_mat = inner_ort_basis(
                vec_ortho)  # .transpose(-1, -2)             #чтобы было согласованно с предсказанием

            r_pred = self.gen(seq.unsqueeze(0), r=r_curr.unsqueeze(0))
            shifts_pred = torch.cat([shifts_pred, r_pred], dim=0)

            # I = torch.ones_like(U)
            # R = ortho_mat.transpose(-1, -2) @ (I + U) @ ortho_mat
            # rad_vec = (ortho_mat.transpose(-1, -2) @ rad_vec.unsqueeze(-1)).squeeze(-1)
            # trans_vec = (ortho_mat.transpose(-1, -2) @ trans_vec.unsqueeze(-1)).squeeze(-1)

            r_pred = ortho_mat.transpose(-1, -2) @ delta + r_curr[-1] + r_pred
            full_cords_pred = torch.cat([full_cords_pred, r_pred], dim=0)

            r_curr = torch.cat([r_curr[1:], r_pred], dim=0)
            cord = torch.cat([cord, r_pred])
        diff = cord.unsqueeze(1) - cord.unsqueeze(0)

        diff_q = self.Q(diff)
        diff_k = self.K(diff).permute(2, 1, 0)
        dr_v = self.V(dr)
        mat = torch.einsum('ijk,kjl->il', diff_q, diff_k)
        d_k = diff_q.shape[-1]
        mat = mat / (d_k ** 0.5)
        attn_weights = torch.softmax(mat, dim=-1)
        df = torch.einsum('ij,jl->il', attn_weights, dr_v)

        df = self.act_fn(df)
        df = self.r_norm(df)

        cord = df + cord

        return diff, cord

