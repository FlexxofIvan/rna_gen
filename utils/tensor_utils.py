import torch

def compute_angles(x: torch.tensor,  epsilon=1e-6):
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

def inner_ort_basis(x, eps=1e-6):
    e1 = x[:, 0] / (x[:, 0].norm(dim=1, keepdim=True) + eps)
    e2 = x[:, 1] - (e1 * x[:, 1]).sum(dim=1, keepdim=True) * e1
    e2 = e2 / (e2.norm(dim=1, keepdim=True) + eps)
    e3 = x[:, 2] - (e1 * x[:, 2]).sum(dim=1, keepdim=True) * e1 - (e2 * x[:, 2]).sum(dim=1, keepdim=True) * e2
    e3 = e3 / (e3.norm(dim=1, keepdim=True) + eps)
    return torch.stack([e1, e2, e3], dim=1)

def mat_mul_vec(mat, vec):
    return (mat @ vec.unsqueeze(-1)).squeeze(-1)


def normalize_basis(a, b, c):

    x = b - a
    x = x / x.norm()

    temp = c - a
    z = torch.cross(x, temp)
    z = z / z.norm()

    y = torch.cross(z, x)

    R = torch.stack([x, y, z], dim=1)  # (3, 3) — матрица поворота
    return R

def loc_basis(v):
    v20 = (v[2] - v[0])
    e20 = v20/torch.norm(v20)

    v10 = (v[1] - v[0])
    v_sch10 = v10 - (torch.dot(v10, e20))*e20
    e10 = v_sch10/torch.norm(v_sch10)
    e_ = torch.cross(e20, e10)/torch.norm(torch.cross(e10, e20))
    return torch.stack([e20, e10, e_], dim=1)