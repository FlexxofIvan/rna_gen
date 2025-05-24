import torch.nn as nn
import torch

from utils.tensor_utils import compute_angles, ortho_basis
import torch.nn.functional as F

import torch.nn.utils as utils



num_emb = 6
num_dehid_pos = 4
pad_idx =5


class Att(nn.Module):
    def __init__(self, h_dq, num_h, h_dv, device='cuda', dropout=0.2):
        super(Att, self).__init__()

        self.h_dq = h_dq
        self.h_dv = h_dv
        self.Q = nn.Linear(h_dq, h_dq*num_h)
        self.K = nn.Linear(h_dq, h_dq*num_h)
        self.V = nn.Linear(h_dv, h_dv * num_h)

        self.h_n = num_h
        self.norm = nn.LayerNorm(h_dv)

        self.proj = nn.Linear(self.h_n * self.h_dv, self.h_dv)
        self.act_fn = nn.GELU()


    def forward(self, q, k, v):
        Q = self.Q(q)
        K = self.K(k)
        V = self.V(v)

        batch_size, seq_len = Q.shape[0], Q.shape[1]

        Q = Q.reshape(batch_size, seq_len, self.h_n, self.h_dq).permute(0, 2, 1, 3)  # [B, h_n, L, h_dq]
        K = K.reshape(batch_size, seq_len, self.h_n, self.h_dq).permute(0, 2, 3, 1)  # [B, h_n, h_dq, L]
        V = V.reshape(batch_size, seq_len, self.h_n, self.h_dv).permute(0, 2, 1, 3)  # [B, h_n, L, h_dv]

        score = torch.einsum('bnij,bnjk->bnik', Q, K) / (self.h_dq ** 0.5)
        score = F.softmax(score, dim=-1)

        att_out = torch.einsum('bnik,bnkl->bnil', score, V)
        att_out = att_out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.h_n * self.h_dv)
        out = self.proj(att_out)
        out = self.act_fn(out)

        out = self.norm(out + v)

        return out


class Block(nn.Module):
    def __init__(self, h_d, num_h, r_dim, device='cuda', dropout=0.2):
        super(Block, self).__init__()
        self.device = device
        self.h_d = h_d

        self.r_dim = r_dim

        self.pos_embedder = nn.Embedding(num_dehid_pos, embedding_dim=h_d)
        self.act = nn.GELU()

        self.nuk_attn = nn.MultiheadAttention(h_d, num_h, batch_first=True)
        self.nuk_attn_seq = nn.MultiheadAttention(h_d, num_h, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(h_d, 2*h_d),
            nn.GELU(),
            nn.Linear(2*h_d, h_d),
            nn.GELU()
        )

        self.norm1 = nn.LayerNorm(h_d)
        self.norm2 = nn.LayerNorm(h_d)
        self.norm3 = nn.LayerNorm(h_d)

        self.norm_att1 = nn.LayerNorm(h_d)
        self.norm_att2 = nn.LayerNorm(h_d)

        self.dropout_attn = nn.Dropout(dropout)



    def forward(self, r, seq):
        B, L, D = seq.shape

        pos_ids = torch.arange(4, device=self.device).unsqueeze(0).repeat(B, 1)
        pos_embeds = self.pos_embedder(pos_ids)         # [B, 4, h_d]

        x = self.norm1(seq + pos_embeds)
        seq = x

        attn_out, _ = self.nuk_attn_seq(x, x, r)
        attn_out = self.dropout_attn(attn_out)
        x = self.norm_att1(r + 0.1 *attn_out)
        x = self.act(x)

        # --- Attention 2 (self-attn) ---
        x2 = self.norm2(x)
        attn_out, _ = self.nuk_attn(x2, x2, x2)
        attn_out = self.dropout_attn(attn_out)
        x = self.norm_att2(x2 + 0.1 *attn_out)
        x = self.act(x)

        # --- Feedforward ---
        #x3 = self.norm3(x)
        x = self.norm3(x + self.ffn(x))

        return seq, x  # [B, 4, h_d]




class Transition(nn.Module):
    def __init__(self, h_d, w_d, device='cuda'):
        super(Transition, self).__init__()

        self.device = device
        self.block = nn.Sequential(

            nn.LayerNorm(h_d),

            nn.Linear(h_d, w_d),
            nn.LeakyReLU(),
            nn.LayerNorm(w_d),

            nn.Linear(w_d, w_d),
            nn.LayerNorm(w_d),
            nn.LeakyReLU(),

            nn.Linear(w_d, h_d),
            nn.LayerNorm(h_d),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)



class Local_module(nn.Module):
    def __init__(self, h_d, num_nuks_head, device='cuda'):
        super(Local_module, self).__init__()

        self.device = device
        self.h_d = h_d

        self.act_fn = nn.GELU()


        self.head_U = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(16),
            self.act_fn,


            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(32),
            nn.Dropout(p=0.1),
            self.act_fn,

            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(16),
            self.act_fn,

            nn.Conv1d(16, 3, kernel_size=1, stride=1),
            nn.InstanceNorm1d(3),
            #self.act_fn
        )

        self.seq_down = nn.Sequential(
            nn.LayerNorm(self.h_d),
            nn.Linear(self.h_d, 64),
            self.act_fn,
            nn.Dropout(p=0.1),
            nn.LayerNorm(64),

            nn.Linear(64, 48),
            self.act_fn,
            nn.LayerNorm(48),

            nn.Linear(48, 36),
            self.act_fn,
            nn.LayerNorm(36),

            nn.Linear(36, 24),
            self.act_fn,
            nn.LayerNorm(24),

            nn.Linear(24, 16),
            self.act_fn,
            nn.LayerNorm(16),

            nn.Linear(16, 8),
            self.act_fn,
            #nn.LayerNorm(8),

            nn.Linear(8, 3),
            #self.act_fn
        )



        self.lin_r = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),  # [B, 3, 3] → [B, 32, 3]
            nn.InstanceNorm1d(16),
            self.act_fn,

            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # [B, 32, 3] → [B, 64, 3]
            nn.Dropout(p=0.1),
            nn.InstanceNorm1d(32),
            self.act_fn,

            nn.Conv1d(32, 16, kernel_size=3, padding=1),  # [B, 64, 3] → [B, 128, 3]
            nn.InstanceNorm1d(16),
            self.act_fn,

            #nn.Dropout(0.1),
            nn.Conv1d(16, 3, kernel_size=1),  # [B, 128, 3] → [B, 3, 3]
            nn.InstanceNorm1d(3),
            self.act_fn,

            nn.AdaptiveAvgPool1d(1)  # [B, 3, 3] → [B, 3, 1]
        )


        self.lin_tr = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),  # [B, 3, 3] → [B, 32, 3]
            nn.InstanceNorm1d(16),
            self.act_fn,

            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # [B, 32, 3] → [B, 64, 3]
            nn.InstanceNorm1d(32),
            self.act_fn,

            nn.Conv1d(32, 16, kernel_size=3, padding=1),  # [B, 64, 3] → [B, 128, 3]
            nn.Dropout(p=0.1),
            nn.InstanceNorm1d(16),
            self.act_fn,

            #nn.Dropout(0.1),
            nn.Conv1d(16, 3, kernel_size=1),  # [B, 128, 3] → [B, 3, 3]
            nn.InstanceNorm1d(3),
            self.act_fn,

            nn.AdaptiveAvgPool1d(1)  # [B, 3, 3] → [B, 3, 1]
        )



        self.out_conv = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, stride=1, padding=1),  # Сначала увеличиваем число фильтров
            nn.InstanceNorm1d(16),
            self.act_fn,

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.05),
            nn.InstanceNorm1d(32),
            self.act_fn,

            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),  # Следующий слой свёртки
            nn.InstanceNorm1d(16),
            self.act_fn,

            #nn.Dropout(0.05),
            nn.Conv1d(16, 3, kernel_size=1, stride=1),  # Сужаем до нужного числа выходных каналов
            nn.InstanceNorm1d(3),
            #self.act_fn
        )



        self.dv_l = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(16),
            self.act_fn,

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.05),
            nn.InstanceNorm1d(32),
            self.act_fn,

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(64),
            self.act_fn,

           # nn.Dropout(0.1),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(32),
            self.act_fn,

            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(16),
            self.act_fn,

            nn.Conv1d(16, 3, kernel_size=1, stride=1),
            nn.InstanceNorm1d(3),
            #self.act_fn
        )

        self.dv_U = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(16),
            self.act_fn,

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.01),
            nn.InstanceNorm1d(32),
            self.act_fn,

            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(16),
            self.act_fn,

            #nn.Dropout(0.1),
            nn.Conv1d(16, 3, kernel_size=1, stride=1),
            nn.InstanceNorm1d(3),
            #self.act_fn
        )

        self.dv_tr = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(16),
            self.act_fn,

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.01),
            nn.InstanceNorm1d(32),
            self.act_fn,

            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(16),
            self.act_fn,

           # nn.Dropout(0.1),
            nn.Conv1d(16, 3, kernel_size=1, stride=1),
            nn.InstanceNorm1d(3),
            self.act_fn,
            nn.AdaptiveAvgPool1d(1)
        )

        self.dv_r = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(16),
            self.act_fn,

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.01),
            nn.InstanceNorm1d(32),
            self.act_fn,

            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(16),
            self.act_fn,

           # nn.Dropout(0.1),
            nn.Conv1d(16, 3, kernel_size=1, stride=1),
            nn.InstanceNorm1d(3),
            self.act_fn,
            nn.AdaptiveAvgPool1d(1)
        )

        self.dr_l = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(16),
            self.act_fn,

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.01),
            nn.InstanceNorm1d(32),
            self.act_fn,

            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(16),
            self.act_fn,

            #nn.Dropout(0.1),
            nn.Conv1d(16, 3, kernel_size=1, stride=1),
            nn.InstanceNorm1d(3),
            self.act_fn,
            nn.AdaptiveAvgPool1d(1)
        )


        self.pos_embed = nn.Embedding(4, self.h_d)

        #self.mean = nn.Parameter(torch.tensor([2.0860, 2.7641, 0.6770]))
        self.mean = nn.Parameter(torch.tensor([1.2, 0.7, 0.9]))


        self.seq_attn_norm1 = nn.LayerNorm(h_d)
        self.seq_attn = nn.MultiheadAttention(embed_dim=h_d, num_heads=4, batch_first=True)
        self.seq_attn_norm2 = nn.LayerNorm(h_d)
        self.seq_attn_dropout = nn.Dropout(0.1)

        self.r_final = nn.Sequential(
            nn.Linear(3, 16),
            self.act_fn,
            nn.LayerNorm(16),

            nn.Linear(16, 32),
            self.act_fn,
            nn.LayerNorm(32),

            nn.Linear(32, 32),
            self.act_fn,
            nn.LayerNorm(32),

            nn.Linear(32, 16),
            self.act_fn,
            nn.LayerNorm(16),

            nn.Linear(16, 3),
        )

    @staticmethod
    def inner_ort_basis(x, eps=1e-3):
        e1 = x[:, 0] / (x[:, 0].norm(dim=1, keepdim=True) + eps)
        e2 = x[:, 1] - (e1 * x[:, 1]).sum(dim=1, keepdim=True) * e1
        e2 = e2 / (e2.norm(dim=1, keepdim=True) + eps)
        e3 = x[:, 2] - (e1 * x[:, 2]).sum(dim=1, keepdim=True) * e1 - (e2 * x[:, 2]).sum(dim=1, keepdim=True) * e2
        e3 = e3 / (e3.norm(dim=1, keepdim=True) + eps)
        return torch.stack([e1, e2, e3], dim=1)  # [B, 3, 3]

    @staticmethod
    def mat_mul(v, mat):
        mv = torch.einsum('ijk, ik -> ij', mat, v)
        return mv


    def forward(self, seq, r=None):
        batch_size = seq.size(0)

        v10 = (r[:, 1, :] - r[:, 0, :]).unsqueeze(1)
        v20 = (r[:, 2, :] - r[:, 0, :]).unsqueeze(1)
        v12 = (r[:, 1, :] - r[:, 0, :]).unsqueeze(1)

        vec_ortho = ortho_basis(v10.squeeze(-2), v20.squeeze(-2))
        r_ortho = self.inner_ort_basis(vec_ortho)


        dr = r - r[:,0].unsqueeze(1)
        dr = torch.einsum('ijk, isk -> isj', r_ortho, dr)


        #dr = self.dr_l(dr.reshape(-1, 9))

        dr = self.dr_l(dr).squeeze(-1) #+ torch.mean(dr, dim=1)

        positions = torch.arange(4, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        pos = self.pos_embed(positions)
        seq = pos + seq

        seq_norm = self.seq_attn_norm1(seq)
        attn_out, _ = self.seq_attn(seq_norm, seq_norm, seq_norm)
        seq = seq + self.seq_attn_dropout(attn_out)
        seq = self.seq_attn_norm2(seq)


        dv = torch.cat((v10, v12, v20), dim=1)
        dv = torch.einsum('ijk, isk -> isj', r_ortho, dv)

        ###здесь ризидуал хорошо работает
        dv = self.dv_l(dv) + dv


        dv_U = self.dv_U(dv)
        dv_tr = self.dv_tr(dv).squeeze(-1)
        dv_r = self.dv_r(dv).squeeze(-1)

        x = self.seq_down(seq)
        out = x

        dout = F.adaptive_avg_pool1d(out.transpose(1,2), 3).transpose(1,2)
        out = self.out_conv(out) + dout

        #ort_U = self.ort_U(r_ortho)
        U =  dv_U  + self.head_U(out)  #+ r_ortho
        I = torch.eye(U.size(-1), device=U.device).expand_as(U)
        U = I + U

        tr_vec = dv_tr + self.lin_tr(out).squeeze(-1)

        #работает с кайфом, не трогать
        mean = self.mean.repeat(batch_size, 1)
        rad_vec = dv_r + self.lin_r(out).squeeze(-1) + mean

        v = self.mat_mul(rad_vec, U)
        vect = v + tr_vec
        vect = vect + dr + mean
        vect = self.r_final(vect) + vect


        return seq, vect



class Global_module(nn.Module):
    def __init__(self, h_d, num_nuks_head, device='cuda'):
        super(Global_module, self).__init__()

        self.device = device

        self.h_d = h_d
        self.num_nuks_head = num_nuks_head

        self.nuk_loc_embedder = nn.Embedding(num_emb, embedding_dim=self.h_d, padding_idx=pad_idx)
        self.loc_curr = Local_module(num_nuks_head=self.num_nuks_head,  h_d=self.h_d)
        self.loc_prev = Local_module(num_nuks_head=self.num_nuks_head,  h_d=self.h_d)
        self.loc_next = Local_module(num_nuks_head=self.num_nuks_head,  h_d=self.h_d)

        self.act_fn = nn.GELU()

        self.pos_embed = nn.Embedding(12, self.h_d)
        self.context_weights = nn.Parameter(torch.randn(3))

        self.convert_layer = nn.Sequential(
            nn.Linear(self.h_d, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, self.h_d),
            #nn.GELU(),
        )


       # self.conv1 = nn.Conv1d(in_channels=4, out_channels=3, kernel_size=1)
        #self.conv2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        self.pos_conv = nn.Conv1d(in_channels=12, out_channels=4, kernel_size=1)


        self.pos_layer = nn.Sequential(
            nn.Linear(self.h_d, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            #nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.LayerNorm(16),
            nn.Linear(16, 3),
            #nn.GELU(),
        )

        self.norm1 = nn.LayerNorm(3)

        self.emb_trans = nn.Sequential(
            nn.Linear(self.h_d, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.LayerNorm(128),
            nn.Linear(128, self.h_d),
            #nn.GELU(),
            #nn.LayerNorm(64),
        )


        self.dropout = nn.Dropout(0.1)

        self.self_attn = nn.MultiheadAttention(embed_dim=self.h_d, num_heads=4, batch_first=True)
        self.dropout_self_attn = nn.Dropout(p=0.1)
        self.norm_self_attn = nn.LayerNorm(self.h_d)

        self.seq_att = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=8, kernel_size=1),  # [B, 12, L] -> [B, 8, L]
            #nn.BatchNorm1d(8),
            nn.GELU(),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=1),  # [B, 8, L] -> [B, 4, L]
            #nn.BatchNorm1d(4),
            #nn.GELU()
        )

        init_weights = torch.tensor([1.0, 0.5, 0.5])
        init_weights = init_weights / init_weights.sum()  # Нормализация
        self.rad_weights = nn.Parameter(init_weights)

        self.rad_conv = nn.Sequential(
            nn.Linear(9, 16),
            self.act_fn,
            nn.LayerNorm(16),

            nn.Linear(16, 9),
            self.act_fn,
            nn.LayerNorm(9),

            nn.Linear(9, 3),
        )



    @staticmethod
    def inner_ort_basis(x, eps=1e-3):
        e1 = x[:, 0] / (x[:, 0].norm(dim=1, keepdim=True) + eps)
        e2 = x[:, 1] - (e1 * x[:, 1]).sum(dim=1, keepdim=True) * e1
        e2 = e2 / (e2.norm(dim=1, keepdim=True) + eps)
        e3 = x[:, 2] - (e1 * x[:, 2]).sum(dim=1, keepdim=True) * e1 - (e2 * x[:, 2]).sum(dim=1, keepdim=True) * e2
        e3 = e3 / (e3.norm(dim=1, keepdim=True) + eps)
        return torch.stack([e1, e2, e3], dim=1)


    @staticmethod
    def mat_mul(v, mat):
        mv = torch.einsum('ijk, ik -> ij', mat, v)
        return mv


    def forward(self, seqs, r):
            batch_size = seqs.shape[0]
            seqs = seqs.reshape(batch_size, 12)

            positions = torch.arange(12, device=seqs.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embed(positions)

            seq = self.nuk_loc_embedder(seqs.long()) + pos_emb
            seq = self.emb_trans(seq) + seq
            #seq = self.dropout(seq)

            seq_prev, seq_curr, seq_next = seq[:, :4], seq[:, 4:8, :], seq[:, 8:, :]
            seq_curr = seq_curr

            seq_stack = torch.stack([seq_prev, seq_curr, seq_next], dim=2)  # [B, 4, 3, h]
            weights = F.softmax(self.context_weights, dim=0)
            seq_cont = (seq_stack * weights.view(1, 1, 3, 1)).sum(dim=2)

            attn_out, _ = self.self_attn(seq, seq, seq)  # [B, 12, h_d]
            attn_out = self.dropout_self_attn(attn_out)
            seq_att = self.norm_self_attn(seq + attn_out)
            seq_att = self.seq_att(seq_att)


            seq_curr = seq_cont + seq_att

            seq_inf = self.convert_layer(seq_curr)
            #seq_inf = self.dropout(seq_inf)

            seq_curr = seq_curr + seq_inf

            seq_new,  rad_vec = self.loc_curr(seq_curr, r=r)
            seq_new_prev, rad_vec_prev = self.loc_prev(seq_prev, r=r)
            seq_new_next, rad_vec_next = self.loc_next(seq_next, r=r)

            #rad_vec = torch.cat([rad_vec_prev, rad_vec, rad_vec_next], dim=1)
            #rad_vec = self.rad_conv(rad_vec)
            rad_vec = self.rad_weights[0]*rad_vec +  self.rad_weights[2]*rad_vec_prev +  self.rad_weights[1]*rad_vec_next

            v10 = (r[:, 1, :] - r[:, 0, :]).unsqueeze(1)
            v20 = (r[:, 2, :] - r[:, 0, :]).unsqueeze(1)

            vec_ortho = ortho_basis(v10.squeeze(-2), v20.squeeze(-2))
            r_ortho = self.inner_ort_basis(vec_ortho)

            rev_mat = r_ortho.transpose(-1, -2)

            rot = self.mat_mul(rad_vec, rev_mat)

            return rot, rad_vec

