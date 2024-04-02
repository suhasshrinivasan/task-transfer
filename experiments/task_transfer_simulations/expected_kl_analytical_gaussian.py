# %%
import torch
from torch.linalg import cholesky

seed = 42
torch.manual_seed(seed)

# %%
tensor_trace = lambda x: x.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


# %%
def expected_kl_gaussian(mu0, mu1, Sigma0, Sigma1, A, Omega):
    B = (Omega @ A).permute(0, -1, -2) @ A
    K0 = (Sigma0.inverse() + B).inverse()
    K1 = (Sigma1.inverse() + B).inverse()

    P_1 = tensor_trace(K1.inverse() @ K0) - k

    M = (A @ (K1 - K0)).permute(0, -1, -2) @ Omega
    N = (K1 @ Sigma1.inverse()) @ mu1.unsqueeze(-1) - (
        K0 @ Sigma0.inverse()
    ) @ mu0.unsqueeze(-1)
    C1 = cholesky(K1.inverse())

    Q = M.permute(0, -1, -2) @ C1 @ C1.permute(0, -1, -2) @ M
    Zeta0 = Omega + A @ (A @ Sigma0).permute(0, -1, -2)
    m0 = A @ mu0.unsqueeze(-1)
    P_2_1 = tensor_trace(Q @ Zeta0) + ((Q @ m0).permute(0, -1, -2) @ m0).squeeze(1)

    P_2_2 = (
        (
            (M.permute(0, -1, -2) @ C1 @ C1.permute(0, -1, -2) @ N).squeeze(-1)
            + (N.permute(0, -1, -2) @ C1 @ C1.permute(0, -1, -2) @ M).squeeze(-2)
        )
        @ m0
    ).squeeze(-1)

    P_2_3 = (N.permute(0, -1, -2) @ C1 @ C1.permute(0, -1, -2) @ N).squeeze(-1)

    P_2 = P_2_1 + P_2_2 + P_2_3

    P_3 = torch.log(torch.det(K1) / torch.det(K0))

    return 0.5 * (P_1 + P_2 + P_3)


# %%
n = 1000
k = 2
start_mu0 = 0
end_mu0 = 6
mu0s = torch.linspace(0, 5, 1000)
mu0_grid1, mu0_grid2 = torch.meshgrid(mu0s, mu0s, indexing="ij")
mu0 = torch.stack([mu0_grid1.flatten(), mu0_grid2.flatten()], dim=1)
# %%
mu1 = torch.tensor([0.0, 0.0]).repeat(n * n, 1)
# %%

Sigma0 = torch.rand(n * n, k, k)
Sigma0 = Sigma0 @ Sigma0.permute(0, -1, -2) + torch.finfo(Sigma0.dtype).eps * torch.eye(
    k
).unsqueeze(0)

Sigma1 = torch.rand(n * n, k, k)
Sigma1 = Sigma1 @ Sigma1.permute(0, -1, -2) + torch.finfo(Sigma1.dtype).eps * torch.eye(
    k
).unsqueeze(0)
Sigma1 = Sigma0
# %%
d = 3
A = torch.rand(n * n, d, k)
Omega = torch.eye(d).unsqueeze(0)
d = A.shape[1]
# %%
kl = expected_kl_gaussian(mu0, mu1, Sigma0, Sigma1, A, Omega)

# %%
