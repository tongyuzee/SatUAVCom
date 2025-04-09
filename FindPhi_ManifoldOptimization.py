import torch
import geoopt
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 系统参数
S = 2
U = 3
N = 4
M = 16
sigma2 = 1e-3

# 生成数据（使用 complex64）
h_su = torch.randn(U, S, N, dtype=torch.complex64)
g_Ru = torch.randn(U, M, dtype=torch.complex64)
H_sR = torch.randn(S, N, M, dtype=torch.complex64)
W_su = torch.randn(U, S, N, dtype=torch.complex64)

def compute_Rsum(Phi):
    SINR_list = []
    for u in range(U):
        signal = torch.tensor(0.0, dtype=torch.complex64, requires_grad=True)
        for s in range(S):
            equiv_channel = h_su[u, s, :] + g_Ru[u, :] @ Phi @ H_sR[s, :, :].T
            signal = signal + torch.sum(equiv_channel.conj() * W_su[u, s, :])
        signal_power = torch.abs(signal) ** 2

        interference_power = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        for u_prime in range(U):
            if u_prime != u:
                interference_sum = torch.tensor(0.0, dtype=torch.complex64, requires_grad=True)
                for s in range(S):
                    equiv_channel = h_su[u, s, :] + g_Ru[u, :] @ Phi @ H_sR[s, :, :].T
                    interference_sum = interference_sum + torch.sum(equiv_channel.conj() * W_su[u_prime, s, :])
                interference_power = interference_power + torch.abs(interference_sum) ** 2

        sinr_u = signal_power / (interference_power + sigma2)
        SINR_list.append(sinr_u)
    
    SINR = torch.stack(SINR_list)
    return torch.sum(torch.log2(1 + SINR))

# 关键修正1：启用梯度追踪的流形张量初始化
Phi_real = torch.randn(M, 2, dtype=torch.float32, requires_grad=True)  # 必须设置 requires_grad=True
Phi_real = geoopt.ManifoldTensor(
    Phi_real,
    manifold=geoopt.manifolds.Sphere()
).requires_grad_(True)  # 显式启用梯度

# 关键修正2：使用正确的优化器参数组
optimizer = geoopt.optim.RiemannianSGD([Phi_real], 0.002)

num_iterations = 2000
R_sum_history = []
for iteration in range(num_iterations):
    optimizer.zero_grad()
    
    # 关键修正3：保留梯度链接的复数转换
    Phi_diag = torch.view_as_complex(Phi_real.contiguous()).type(torch.complex64)
    Phi = torch.diag(Phi_diag)
    
    R_sum = compute_Rsum(Phi)
    R_sum_history.append(R_sum.item())
    
    # 反向传播时分离梯度检查
    (-R_sum).backward(retain_graph=False)  # 确保无梯度残留
    optimizer.step()

    if iteration % 100 == 0:
        print(f"Iteration {iteration}: R_sum = {R_sum.item():.4f}")

# 结果验证
optimal_Phi = torch.diag(torch.view_as_complex(Phi_real.detach()))
print("\nOptimal Phi magnitudes:", torch.abs(optimal_Phi.diagonal()))  # 应全为1.0

plt.plot(R_sum_history)
plt.xlabel("Iteration")
plt.ylabel("Sum Rate")
plt.show()