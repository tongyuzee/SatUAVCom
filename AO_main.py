import numpy as np
import torch
import matplotlib.pyplot as plt
import RISSatUAVCom
import FindW_WMMSE
import FindPhi_GradientAscent

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_Sinr_Rsum(U, S, N, M, h_su, H_sR, g_Ru, w_su, theta):
    # 构造 Phi 矩阵：Phi = diag(e^{j*theta})
    exp_j_theta = np.exp(1j * theta)
    Phi = np.diag(exp_j_theta)  # (M, M) 的对角矩阵

    SINR = np.zeros(U)
    sigout = np.zeros(U)
    for u in range(U):
        # 信号部分 (分子)
        signal = 0
        for s in range(S):
            # 等效信道: h_{s,u}^T + G_{R,u}^T Phi H_{s,R}^T
            # 注意 h_su 的索引变为 [u, s, :] 因为形状改为 (U, S, N)
            equiv_channel = h_su[u, s, :]+ g_Ru[u, :]@ Phi @ H_sR[s, :, :].T
            signal += np.vdot(equiv_channel , w_su[u, s, :])
        signal_power = np.abs(signal) ** 2

        sigout[u] = signal

        # 干扰部分 (分母)
        interference_power = 0
        for u_prime in range(U):
            if u_prime != u:
                interf = 0
                for s in range(S):
                    equiv_channel = h_su[u, s, :] + g_Ru[u, :] @ Phi @ H_sR[s, :, :].T
                    interf += np.vdot(equiv_channel, w_su[u_prime, s, :])
                interference_power += np.abs(interf) ** 2
        # SINR
        SINR[u] = signal_power / (interference_power + sigma2)
    # 计算 R_sum
    R_sum = np.sum(np.log2(1 + SINR))
    return SINR, R_sum, sigout

def compute_sum_rate(U, S, N, M, H, W, sigma2):
    """计算和速率 R"""
    R = 0
    for u in range(U):
        signal = np.vdot(H[:, u], W[:, u]) * np.vdot(W[:, u], H[:, u])
        interfere = 0
        for u_prime in range(U):
            if u_prime != u:
                interfere += np.vdot(H[:, u], W[:, u_prime]) * np.vdot(W[:, u_prime], H[:, u])
        INR = sigma2 + interfere
        R += np.log2(1 + (signal / INR))
    return np.abs(R)

set_seed(42)

S = 2  # 卫星数量
U = 3  # 无人机数量
N = 4  # 天线数量
M = 16  # RIS单元数量
P_s = 1  # 发射功率
sigma2 = 1e-18  # 噪声方差
gain_factor = 1e8  # 增益因子

# 生成信道矩阵
Sat_UAV_comm = RISSatUAVCom.RISSatUAVCom(10, U, S, N, M)
h_su, H_sR, g_Ru = Sat_UAV_comm.setup_channel()

# 在通信系统中，信号功率通常是∣h^H w∣^2 
h_su = np.conj(h_su)
H_sR = np.conj(H_sR)
g_Ru = np.conj(g_Ru)

# 信道矩阵放大n倍，噪声功率放大n^2倍，SINR不变，优化结果不变。
h_su = h_su * gain_factor
H_sR = H_sR * gain_factor
sigma2 = sigma2 * gain_factor ** 2

theta = np.random.uniform(0, 2 * np.pi, M) # 随机初始化theta

Rate = [0]

for iter in range(1000):

    Phi = np.diag(np.exp(1j * theta))  # size: M x M
    # 计算等效信道矩阵 H ， size: S*N x U
    H = np.zeros((S * N, U), dtype=complex)
    for u in range(U):
        for s in range(S):
            h_tilde = h_su[u, s, :].T + g_Ru[u, :] @ Phi @ H_sR[s, :, :].T
            H[s * N:(s + 1) * N, u] = h_tilde

    if iter == 0:
        # 基于 MRT 的预编码矩阵 W 初始化， size: S*N x U
        W = np.zeros((S * N, U), dtype=complex)
        for i in range(U):
            h_i = H[:, i]  # 第 i 个用户的信道向量
            w_i = h_i / np.linalg.norm(h_i, 2)  # 归一化
            W[:, i] = w_i * np.sqrt(P_s / U)  # 分配功率
    else:
        W = W_opt

    # W_su = np.zeros((U, S, N), dtype=complex)
    # for u in range(U):
    #     for s in range(S):
    #         W_su[u, s, :] = W[s * N:(s + 1) * N, u]

    # S1, R1 = compute_Sinr_Rsum(U, S, N, M, h_su, H_sR, g_Ru, W_su, theta)
    # R2 = compute_sum_rate(U, S, N, M, H, W, sigma2)

    # WMMSE算法求解最优预编码矩阵
    Woptimization = FindW_WMMSE.FindW_WMMSE(S, U, N, M, H, W, P_s, sigma2)
    W_opt, rate_w = Woptimization.optimize()
    W_su = np.zeros((U, S, N), dtype=complex)
    for u in range(U):
        for s in range(S):
            W_su[u, s, :] = W_opt[s * N:(s + 1) * N, u]
    # Woptimization.plot_rate(rate_w)
    S_w, R_w, sigout_w= compute_Sinr_Rsum(U, S, N, M, h_su, H_sR, g_Ru, W_su, theta)
    Rate.append(R_w)

    # 梯度上升算法求解最优RIS矩阵
    h_su_t = torch.tensor(h_su, dtype=torch.complex128).clone().detach()
    H_sR_t = torch.tensor(H_sR, dtype=torch.complex128).clone().detach()
    g_Ru_t = torch.tensor(g_Ru, dtype=torch.complex128).clone().detach()
    W_su_t = torch.tensor(W_su, dtype=torch.complex128).clone().detach()
    theta_t = torch.tensor(theta, dtype=torch.float64).clone().detach()

    # R_init = rate_w[-1]
    R_init = 0
    PhiOptimization = FindPhi_GradientAscent.FindPhi_GA(S, U, N, M, h_su_t, H_sR_t, g_Ru_t, W_su_t, theta_t, R_init, sigma2)
    theta, rate_phi = PhiOptimization.optimize_theta(2000, 0.01)
    # PhiOptimization.plot_results(rate_phi)
    S_phi, R_phi, sigout_phi = compute_Sinr_Rsum(U, S, N, M, h_su, H_sR, g_Ru, W_su, theta)
    Rate.append(R_phi)

    print(f'迭代次数: {iter}, FindW: rate_w={rate_w[-1]}, FindPhi: rate_phi={rate_phi[-1]}')

    if iter > 0 and abs(Rate[-1] - Rate[-2]) < 1e-4 and abs(Rate[-2] - Rate[-3]) < 1e-3:
        break

plt.figure(figsize=(10, 6))
plt.plot(Rate)
plt.ylabel('Sum Rate') 
plt.xlabel('iterations')
plt.grid(True)
plt.show()