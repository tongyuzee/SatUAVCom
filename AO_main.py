import numpy as np
import matplotlib.pyplot as plt
import RISSatUAVCom
import FindW_WMMSE


S = 2  # 卫星数量
U = 3  # 无人机数量
N = 4  # 天线数量
M = 16  # RIS单元数量
P_s = 1  # 发射功率
sigma2 = 1e-16  # 噪声方差

# 生成信道矩阵
Sat_UAV_comm = RISSatUAVCom.RISSatUAVCom(10, U, S, N, M)
h_su, H_sR, g_Ru = Sat_UAV_comm.setup_channel()

# 初始化RIS矩阵 Phi, size: M x M
Phi = np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, M)))  

# 计算等效信道矩阵 H ， size: S*N x U
H = np.zeros((S * N, U), dtype=complex)
for u in range(U):
    for s in range(S):
        h_tilde = h_su[u, s, :].T + g_Ru[u, :] @ Phi @ H_sR[s, :, :].T
        H[s * N:(s + 1) * N, u] = h_tilde

# 信道矩阵放大n倍，噪声功率放大n^2倍，SINR不变，优化结果不变。
H_up = H * 1e6  # 放大信道矩阵，避免数值问题
sigma2 = sigma2 * 1e12  # 放大噪声功率，避免数值问题

# 初始化预编码矩阵 W 并归一化， size: S*N x U
W_init = H_up / np.linalg.norm(H_up, 'fro') * np.sqrt(P_s)

# WMMSE算法求解最优预编码矩阵
optimizer = FindW_WMMSE.FindW_WMMSE(S, U, N, M, H_up, W_init, P_s, sigma2)
W_opt, rate = optimizer.optimize()
optimizer.plot_rate(rate)