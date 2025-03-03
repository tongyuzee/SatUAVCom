import numpy as np

# 设置系统参数
S = 2  # LEO 卫星数量
U = 3  # UAV 数量
N = 4  # 天线数量
M = 16  # RIS 反射单元数
P_max = 10  # 最大功率约束
noise_power = 1e-3  # 噪声功率

# 随机初始化信道矩阵
np.random.seed(42)
h_s_u = np.random.randn(S, U, N) + 1j * np.random.randn(S, U, N)  # LEO到UAV信道 NX1
H_s_R = np.random.randn(S, N, M) + 1j * np.random.randn(S, N, M)  # LEO到RIS信道 NXM
G_R_u = np.random.randn(U, M) + 1j * np.random.randn(U, M)  # RIS到UAV信道 MX1
Phi = np.diag(np.exp(1j * np.random.uniform(0, 2*np.pi, M)))  # RIS相移矩阵 M*M

# 初始化波束成形向量
W = np.random.randn(S, U, N) + 1j * np.random.randn(S, U, N)

# 计算信道增益
def compute_channel_gain(W, h_s_u, H_s_R, G_R_u, Phi):
    SINR = np.zeros(U)
    for u in range(U):
        signal_power = 0
        interference_power = 0
        for s in range(S):
            effective_channel = h_s_u[s, u].T + G_R_u[u].T @ Phi @ H_s_R[s].T
            signal_power += np.abs(effective_channel @ W[s, u]) ** 2
        
        for u_prime in range(U):
            if u_prime != u:
                for s in range(S):
                    effective_channel = h_s_u[s, u].T + G_R_u[u].T @ Phi @ H_s_R[s].T
                    interference_power += np.abs(effective_channel @ W[s, u_prime]) ** 2
        
        SINR[u] = signal_power / (interference_power + noise_power)
    
    return SINR

# WMMSE 交替优化过程
def WMMSE_optimization(W, H_s_u, H_s_R, G_R_u, Phi, max_iter=100, tol=1e-4):
    U, S, N = W.shape[1], W.shape[0], W.shape[2]
    
    for iteration in range(max_iter):
        # Step 1: 更新均衡器 g_u
        G = np.zeros(U, dtype=complex)
        for u in range(U):
            signal = 0
            for s in range(S):
                effective_channel = H_s_u[s, u] + G_R_u[u] @ Phi @ H_s_R[s].T
                signal += effective_channel @ W[s, u]
            G[u] = 1 / signal if np.abs(signal) > 0 else 0
        
        # Step 2: 更新权重参数 λ_u
        Lambda = np.zeros(U)
        for u in range(U):
            error = 1 - np.abs(G[u])**2 * compute_channel_gain(W, H_s_u, H_s_R, G_R_u, Phi)[u]
            Lambda[u] = 1 / (error + 1e-5)  # 避免除零
        
        # Step 3: 更新波束成形向量 W
        for s in range(S):
            for u in range(U):
                effective_channel = H_s_u[s, u] + G_R_u[u] @ Phi @ H_s_R[s].T
                W[s, u] = Lambda[u] * G[u] * effective_channel.conj().T
                # 归一化功率
                W[s, u] /= np.linalg.norm(W[s, u]) * np.sqrt(P_max)
        
        # 计算新的 SINR
        SINR = compute_channel_gain(W, H_s_u, H_s_R, G_R_u, Phi)
        R_sum = np.sum(np.log2(1 + SINR))
        
        print(f"Iter {iteration+1}: R_sum = {R_sum:.3f} bps/Hz")
        # 检查收敛性
        if iteration > 0 and np.abs(R_sum - prev_R_sum) < tol:
            break
        
        prev_R_sum = R_sum

        
    
    return W, R_sum

# 执行 WMMSE 优化
W_opt, R_sum_opt = WMMSE_optimization(W, h_s_u, H_s_R, G_R_u, Phi)

# 输出最终优化的遍历速率
print(f"优化后的总遍历速率: {R_sum_opt:.4f} bit/s/Hz")
