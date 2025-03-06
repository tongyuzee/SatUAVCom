import numpy as np
import matplotlib.pyplot as plt
import RISSatUAVCom
import FindW_WMMSE


S = 2  # 卫星数量
U = 3  # 无人机数量
N = 4  # 天线数量
M = 5  # RIS单元数量
P_s = 1  # 发射功率
sigma2 = 1e-4  # 噪声方差

# 生成信道矩阵
Sat_UAV_comm = RISSatUAVCom(10, U, S, N, M)
h_su, H_sR, g_Ru = Sat_UAV_comm.setup_channel()

# 初始化预编码矩阵 W 并归一化， size: S*N x U
W_init = np.random.randn(S * N, U) + 1j * np.random.randn(S * N, U)
total_power = np.trace(W_init @ W_init.conj().T)
W_init = W_init * np.sqrt(P_s / total_power)
# 初始化RIS矩阵 Phi, size: M x M
Phi = np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, M)))  

# WMMSE算法求解最优预编码矩阵
optimizer = FindW_WMMSE(S, U, N, M, h_su, H_sR, g_Ru, W_init, Phi, P_s, sigma2)
W_opt, rate = optimizer.optimize()