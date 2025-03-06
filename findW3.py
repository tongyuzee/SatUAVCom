import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 系统参数
S = 2  # 卫星数量
U = 3  # 无人机数量
N = 4  # 天线数量
M = 5  # RIS单元数量
P_s = 10.0
sigma2 = 1e-4
gain_factor = 20.0

# 随机初始化信道矩阵
np.random.seed(42)
h_su = np.random.randn(U, S, N) + 1j * np.random.randn(U, S, N)  # LEO到UAV信道 NX1
H_sR = np.random.randn(S, N, M) + 1j * np.random.randn(S, N, M)  # LEO到RIS信道 NXM
G_Ru = np.random.randn(U, M) + 1j * np.random.randn(U, M)  # RIS到UAV信道 MX1
Phi = np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, M)))  # RIS相移矩阵 M*M

H = np.zeros((S*N, U), dtype=complex)  # 等效信道矩阵
for u in range(U):
    for s in range(S):
        h_tilde = h_su[u, s, :].T + G_Ru[u, :].T @ Phi @ H_sR[s, :, :].T  # 1XN
        H[s*N:(s+1)*N, u] = h_tilde

# 初始化
W = np.random.randn(S*N, U) + 1j * np.random.randn(S*N, U)  # 随机初始化
total_power = np.trace(W @ W.conj().T)  # 计算总功率
W = W * np.sqrt(P_s / total_power)  # 归一化

# SINR和速率计算
def compute_sum_rate(H, W, sigma2, U):
    """Calculate the sum rate R."""
    sum1 = 0
    for i in range(U):
        sum1 += np.vdot(H[:, i], W[:, i]) * np.vdot(W[:, i], H[:, i])
    interfere = np.zeros(U)
    R = 0
    for i in range(U):
        interfere[i] = sum1 - np.vdot(H[:, i], W[:, i]) * np.vdot(W[:, i], H[:, i])
        INR = sigma2 + interfere[i]  # Interference plus noise
        sinal = np.vdot(H[:, i], W[:, i]) * np.vdot(W[:, i], H[:, i])  # Desired signal
        R += np.log2(1 + (sinal / INR))  # Rate for user i
    return R

def generate_G(H, W, sigma2, U):
    """Generate receive filters G."""
    G = np.zeros(U, dtype=complex)
    sum1 = 0
    for i in range(U):
        sum1 += np.vdot(H[:, i], W[:, i]) * np.vdot(W[:, i], H[:, i])
    for i in range(U):
        G[i] = np.vdot(H[:, i], W[:, i]) / (sum1 + sigma2)
    return G

def generate_La(H, W, sigma2, U):
    """Generate weights Lambda."""
    La = np.zeros(U)
    sum1 = 0
    for i in range(U):
        sum1 += np.vdot(H[:, i], W[:, i]) * np.vdot(W[:, i], H[:, i])
    for i in range(U):
        temp = np.vdot(W[:, i], H[:, i]) * (1 / (sum1 + sigma2)) * np.vdot(H[:, i], W[:, i])
        La[i] = 1 / (1 - temp)
    return La

def generate_W(H, G, La, S_N, U, P_s):
    """Generate precoding vectors W."""
    sum2 = np.zeros((S_N, S_N), dtype=complex)
    for i in range(U):
        sum2 += np.outer(H[:, i], H[:, i].conj()) * G[i] * La[i] * np.conj(G[i])
    
    # Binary search for optimal mu
    mu_max = 10
    mu_min = 0
    iter_max = 100
    
    for iter in range(iter_max):
        mu = (mu_min + mu_max) / 2
        Pt = 0
        W_opt = np.zeros((S_N, U), dtype=complex)
        for i in range(U):
            A = sum2 + mu * np.eye(S_N)
            # b = H[:, i] * G[i] * La[i] 
            W_opt[:, i] = np.linalg.solve(A, H[:, i])* G[i] * La[i]
            Pt += np.real(np.trace(np.outer(W_opt[:, i], W_opt[:, i].conj())))
        if Pt > P_s:
            mu_min = mu
        else:
            mu_max = mu
        if abs(mu_max - mu_min) < 1e-5 :
            break
    print(f'求解最优mu共迭代{iter}次,mu*={mu},P={Pt}')
    return W_opt

# WMMSE算法
# Iterative Algorithm
rate = []
for iter in range(200):
    R_pre = compute_sum_rate(H, W, sigma2, U)  # Previous sum rate
    rate.append(R_pre)
    G = generate_G(H, W, sigma2, U)  # Update receive filters
    La = generate_La(H, W, sigma2, U)  # Update weights
    W = generate_W(H, G, La, S*N, U, P_s)  # Update precoding vectors
    R = compute_sum_rate(H, W, sigma2, U)  # Current sum rate
    if abs(R - R_pre) < 1e-5:  # Convergence check
        break
print(f'求解和速率共迭代{iter}次')
rate.append(R)  # Append final rate


# Plot Results
plt.plot(rate)
plt.ylabel('Sum Rate')
plt.xlabel('iterations')
plt.show()