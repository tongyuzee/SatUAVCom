import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 系统参数
S = 2  # LEO 卫星数量
U = 3  # UAV 数量
N = 4  # 每颗 LEO 卫星的天线数量
M = 8  # RIS 反射单元数量
sigma2 = 1e-3  # 噪声功率

# 随机生成信道矩阵和变量（使用 PyTorch 的张量）
# h_{s,u}: S x U x N (相当于转置后的 1 x N)
h_su = torch.randn(S, U, N, dtype=torch.complex64)
# G_{R,u}: U x M (相当于转置后的 1 x M)
G_Ru = torch.randn(U, M, dtype=torch.complex64)
# H_{s,R}: S x N x M (转置后为 M x N)
H_sR = torch.randn(S, N, M, dtype=torch.complex64)
# w_{s,u}: S x U x N
w_su = torch.randn(S, U, N, dtype=torch.complex64)
# s_{s,u}: S x U
# s_su = torch.randn(S, U, dtype=torch.complex64)

# 将 sigma2 转换为张量
sigma2 = torch.tensor(sigma2, dtype=torch.float32)

# 计算 SINR 和 R_sum 的函数
def compute_Rsum(theta, h_su, G_Ru, H_sR, w_su, sigma2, S, U, M):
    # 构造 Phi 矩阵：Phi = diag(e^{j*theta})
    # theta 是 (M,) 的向量，e^{j*theta} 是 (M,) 的复数向量
    exp_j_theta = torch.exp(1j * theta)  # e^{j*theta}
    Phi = torch.diag(exp_j_theta)  # (M, M) 的对角矩阵

    SINR = torch.zeros(U, dtype=torch.float32)
    for u in range(U):
        # 信号部分 (分子)
        signal = torch.tensor(0.0, dtype=torch.complex64)
        for s in range(S):
            # 等效信道: h_{s,u}^T + G_{R,u}^T Phi H_{s,R}^T
            # G_{R,u}^T: (1, M), Phi: (M, M), H_{s,R}^T: (M, N) -> (1, N)
            equiv_channel = h_su[s, u, :].T + G_Ru[u, :].T @ Phi @ H_sR[s, :, :].T
            # w_{s,u} * s_{s,u}: (N,) * scalar -> (N,)
            signal += equiv_channel @ w_su[s, u, :]
        signal_power = torch.abs(signal) ** 2

        # 干扰部分 (分母)
        interference = torch.tensor(0.0, dtype=torch.complex64)
        for s in range(S):
            # 对所有 u' != u 的干扰
            interference_sum = torch.zeros(N, dtype=torch.complex64)
            for u_prime in range(U):
                if u_prime != u:
                    interference_sum += w_su[s, u_prime, :]
            equiv_channel = h_su[s, u, :] + G_Ru[u, :] @ Phi @ H_sR[s, :, :].T
            interference += equiv_channel @ interference_sum
        interference_power = torch.abs(interference) ** 2

        # SINR
        SINR[u] = signal_power / (interference_power + sigma2)

    # 计算 R_sum
    R_sum = torch.sum(torch.log2(1 + SINR))
    return R_sum

# 梯度下降优化（使用 PyTorch 的优化器）
def optimize_theta(h_su, G_Ru, H_sR, w_su, sigma2, S, U, M, max_iter=200, learning_rate=0.01):
    # 初始化 theta 为可优化的参数，并启用梯度计算
    theta = torch.tensor(np.random.uniform(0, 2 * np.pi, M), dtype=torch.float32, requires_grad=True)
    # 使用 Adam 优化器
    optimizer = torch.optim.Adam([theta], lr=learning_rate)
    # 记录 R_sum 的历史
    R_sum_history = []
    for iteration in range(max_iter):
        # 清零梯度
        optimizer.zero_grad()
        # 计算 R_sum（前向传播）
        R_sum = compute_Rsum(theta, h_su, G_Ru, H_sR, w_su, sigma2, S, U, M)

        # 反向传播，计算梯度
        (-R_sum).backward()  # 因为 PyTorch 默认是求最小值，所以取负值以求最大值

        # 更新 theta
        optimizer.step()

        # 确保 theta 在 [0, 2pi) 范围内
        with torch.no_grad():
            theta.copy_(theta % (2 * torch.pi))

        # 记录 R_sum
        R_sum_history.append(R_sum.item())

        # 打印当前迭代信息
        if (iteration + 1) % 20 == 0:
            print(f"Iteration {iteration + 1}/{max_iter}, R_sum = {R_sum.item():.4f}")

    return theta.detach().numpy(), R_sum_history

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 设置随机种子
set_seed(42)

# 运行优化
max_iter = 2000
learning_rate = 0.01
theta_opt, R_sum_history = optimize_theta(h_su, G_Ru, H_sR, w_su, sigma2, S, U, M, max_iter, learning_rate)

# 打印最终结果
final_Rsum = compute_Rsum(torch.tensor(theta_opt, dtype=torch.float32), h_su, G_Ru, H_sR, w_su, sigma2, S, U, M)
print(f"\nOptimized R_sum: {final_Rsum.item():.4f}")
print("Optimized theta (in radians):")
print(theta_opt)

# 绘制 R_sum 随迭代次数的变化
plt.figure(figsize=(8, 6))
plt.plot(R_sum_history, label="R_sum")
plt.xlabel("Iteration")
plt.ylabel("Sum Rate (R_sum)")
plt.title("Sum Rate vs. Iteration (PyTorch Optimization)")
plt.grid(True)
plt.legend()
plt.show()